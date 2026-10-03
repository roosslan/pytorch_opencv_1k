"""Командная строка: RTSP -> YOLOv8 + трекер -> захват и сопровождение целей -> ClickHouse."""

import argparse
import importlib.metadata
import logging
import sys
import time
import uuid
from pathlib import Path

from rectrack import storage
from rectrack.capture import LatestFrameReader, make_rtsp_capture
from rectrack.config import Config, ConfigError, ModelConfig, load_config, redact_url
from rectrack.detector import YoloTracker
from rectrack.overlay import annotated, draw_tracks
from rectrack.pipeline import TrackingWorker
from rectrack.proxy import ProxyTunnel, local_url, rtsp_target
from rectrack.recorder import TargetRecorder
from rectrack.snapshots import SnapshotWriter
from rectrack.tracks import TrackRegistry

log = logging.getLogger("rectrack")

WINDOW = "rec_pytorch"
NO_SIGNAL_AFTER_SEC = 3.0


def build_parser() -> argparse.ArgumentParser:
    """Создаёт разбор аргументов командной строки. Описывает ключи:
      --config PATH  - путь к файлу конфигурации TOML (по умолчанию config.toml);
      --headless     - работать без окна с видео (для сервера или службы);
      --self-test    - проверить, что библиотеки и модель загружаются, и выйти;
      --init-db      - создать базу и таблицы в ClickHouse и выйти;
      --list [N]     - показать последние N событий из ClickHouse (по умолчанию 20) и выйти.

    Returns:
        готовый argparse.ArgumentParser; сам разбор аргументов делает вызывающий код.
    """
    parser = argparse.ArgumentParser(prog="rectrack", description=__doc__)
    parser.add_argument("--config", default="config.toml", help="путь к конфигурации (TOML)")
    parser.add_argument("--headless", action="store_true", help="без окна с видео")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить, что PyTorch, ultralytics и модель загружаются, и выйти",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="создать базу и таблицы в ClickHouse и выйти",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="показать последние N событий из ClickHouse (по умолчанию 20) и выйти",
    )
    return parser


def run_self_test(model_cfg: ModelConfig) -> None:
    """Самопроверка установки. Импортирует OpenCV, numpy, PyTorch и ultralytics, пишет в журнал
    их версии (и версию clickhouse-connect) и доступна ли CUDA. Затем загружает модель и
    прогоняет её вместе с трекером по одному чёрному кадру 640x480. Любая проблема (нет
    библиотеки, нет CUDA при device = "cuda", испорченные веса) выбрасывается как исключение.

    Args:
        model_cfg: настройки модели (раздел [model]): какие веса загрузить, на каком устройстве
            запускать, какие классы искать, какой трекер использовать.

    Returns:
        None. Успехом считается отсутствие исключения.
    """
    import cv2
    import numpy
    import torch
    import ultralytics

    log.info(
        "torch %s, ultralytics %s, opencv %s, clickhouse-connect %s",
        torch.__version__,
        ultralytics.__version__,
        cv2.__version__,
        importlib.metadata.version("clickhouse-connect"),
    )
    log.info("CUDA доступна: %s", torch.cuda.is_available())
    detector = YoloTracker(model_cfg)
    detector.load()
    detector.track(numpy.zeros((480, 640, 3), numpy.uint8))


def main(argv: list[str] | None = None) -> int:
    """Точка входа программы. Разбирает аргументы, настраивает журнал (уровень INFO, формат
    «время уровень сообщение») и выбирает режим работы:
      --self-test         - самопроверка; config.toml не обязателен, без него проверяется
                            модель с настройками по умолчанию;
      --init-db / --list  - одна служебная команда к ClickHouse (см. database_command);
      без этих ключей     - основной режим: сопровождение целей с камеры (см. run).

    Args:
        argv: аргументы командной строки без имени программы. None (по умолчанию) означает «взять из
            sys.argv»; явный список удобен в тестах.

    Returns:
        код завершения процесса: 0 - успех; 1 - ошибка работы (самопроверка не пройдена, модель не
        загрузилась, ClickHouse недоступен); 2 - ошибка в конфигурации.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    config_path = Path(args.config)

    if args.self_test:
        # Конфигурация здесь необязательна: без неё проверяется модель по умолчанию.
        try:
            model_cfg = load_config(config_path).model if config_path.exists() else ModelConfig()
            run_self_test(model_cfg)
        except Exception:
            log.exception("Самопроверка не пройдена")
            return 1
        log.info("Самопроверка пройдена")
        return 0

    try:
        config = load_config(config_path)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if args.init_db or args.list is not None:
        return database_command(config, init=args.init_db, count=args.list)
    return run(config, headless=args.headless)


def database_command(config: Config, init: bool, count: int | None) -> int:
    """Выполняет одну служебную команду к ClickHouse и завершает работу.
    Если init истинен, создаёт базу и таблицы по schema.sql (когда их ещё нет) и печатает
    сообщение об успехе. Иначе печатает последние события из таблицы events, новые сверху:
    время (UTC), камера, тип события, класс, номер цели, для 'lost' - сколько цель была
    в кадре, и путь к снимку.

    Args:
        config: полная конфигурация; используется только раздел [clickhouse] (адрес, учётные данные,
            имя базы).
        init: True: создать схему (ключ --init-db); False: показать события (ключ --list).
        count: сколько последних событий показать; None и 0 заменяются на 20. При init=True не
            используется.

    Returns:
        0 при успехе, 1 если ClickHouse недоступен или запрос завершился ошибкой.
    """
    ch = config.clickhouse
    try:
        client = storage.default_client_factory(ch)
        if init:
            storage.ensure_schema(client, ch.database)
            print(f"База '{ch.database}' и таблицы готовы ({ch.host}:{ch.port})")
            return 0
        rows = storage.recent_events(client, ch.database, count or 20)
    except Exception as exc:
        log.error("ClickHouse недоступен (%s:%s): %s", ch.host, ch.port, exc)
        return 1
    if not rows:
        print(f"В {ch.database}.events пока нет записей")
    for ts, event, camera, class_name, track_id, duration, snapshot in rows:
        extra = f", {duration:.1f} с" if event == "lost" else ""
        when = f"{ts:%Y-%m-%d %H:%M:%S}"
        print(f"{when}  {camera}  {event:8} {class_name} #{track_id}{extra}  {snapshot}")
    return 0


def run(config: Config, headless: bool) -> int:
    """Основной режим работы. Загружает модель, запускает запись в ClickHouse (если она включена)
    и собирает цепочку обработки:
      чтение RTSP-потока (LatestFrameReader)
      -> детектор с трекером в фоновом потоке (TrackingWorker)
      -> реестр целей (TrackRegistry)
      -> обработчик событий (TargetRecorder: снимки, журнал, ClickHouse).
    Если в [camera] задан proxy, перед чтением потока поднимает локальный туннель через
    HTTP-прокси (ProxyTunnel), и FFmpeg подключается к камере через него.
    Затем ждёт, пока пользователь закроет окно, нажмёт q или Ctrl+C. При выходе останавливает
    поток обработки, закрывает все сопровождаемые цели с причиной 'shutdown', дописывает
    остаток буфера в ClickHouse и отключается от камеры.

    Args:
        config: полная загруженная и проверенная конфигурация (все разделы).
        headless: True: работать без окна, остановка только по Ctrl+C; False: показывать окно с
            видео и нарисованными целями.

    Returns:
        0 после штатной остановки, 1 если не удалось загрузить модель.
    """
    detector = YoloTracker(config.model)
    try:
        started = time.monotonic()
        detector.load()
        log.info("Модель загружена за %.1f с", time.monotonic() - started)
    except Exception:
        log.exception("Не удалось загрузить модель")
        return 1

    ch, snaps, trk, cam = config.clickhouse, config.snapshots, config.tracking, config.camera

    sink = None
    if ch.enabled:
        sink = storage.ClickHouseSink(ch)
        sink.start()
        log.info("ClickHouse: %s:%s, база '%s'", ch.host, ch.port, ch.database)
    else:
        log.info("Запись в ClickHouse отключена, события идут только в журнал")

    recorder = TargetRecorder(
        camera=cam.name,
        session_id=uuid.uuid4(),
        sink=sink,
        snapshots=SnapshotWriter(snaps.directory) if snaps.enabled else None,
        annotate=annotated if snaps.annotate else None,
    )
    registry = TrackRegistry(
        on_captured=recorder.on_captured,
        on_point=recorder.on_point,
        on_lost=recorder.on_lost,
        confirm_hits=trk.confirm_hits,
        lost_timeout=trk.lost_timeout_sec,
        points_interval=trk.points_interval_sec,
        trail_length=trk.trail_length,
    )

    log.info("Камера '%s': %s (%s)", cam.name, redact_url(cam.rtsp_url), cam.transport)
    url, tunnel = cam.rtsp_url, None
    if cam.proxy:
        host, port = rtsp_target(cam.rtsp_url)
        tunnel = ProxyTunnel(cam.proxy, host, port, timeout=cam.timeout_sec)
        tunnel.start()
        url = local_url(cam.rtsp_url, tunnel.port)
        log.info("Через прокси %s (локальный туннель %s)", redact_url(cam.proxy), redact_url(url))
    reader = LatestFrameReader(
        lambda: make_rtsp_capture(url, cam.transport, cam.timeout_sec),
        cam.reconnect_delay_sec,
    )
    worker = TrackingWorker(
        reader,
        detector,
        registry,
        reset_gap=trk.reset_gap_sec,
        stats_interval=trk.stats_interval_sec,
    )

    reader.start()
    worker.start()
    try:
        if headless:
            run_headless()
        else:
            run_viewer(reader, registry)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        registry.close_all("shutdown")  # итог по целям, которые ещё сопровождались
        if sink is not None:
            sink.stop()
        reader.stop()
        if tunnel is not None:
            tunnel.stop()
    return 0


def run_headless() -> None:
    """Режим без окна: держит главный поток живым, пока вся работа идёт в фоновых потоках.
    Цикл бесконечный; прерывается только исключением KeyboardInterrupt (Ctrl+C), которое
    перехватывает вызывающая функция run.

    Returns:
        управление не возвращает, выход только через исключение.
    """
    log.info("Работаю без окна, остановка: Ctrl+C")
    while True:
        time.sleep(1.0)


def run_viewer(reader: LatestFrameReader, registry: TrackRegistry) -> None:
    """Режим с окном: показывает свежие кадры с камеры с нарисованными целями (рамки, номера,
    «хвосты» траекторий). Рисует на копии кадра, потому что тот же кадр в это время читает
    поток детекции. Если новых кадров нет дольше NO_SIGNAL_AFTER_SEC секунд, поверх последнего
    кадра пишется красное «NO SIGNAL». Выходит по клавише q или когда окно закрыто крестиком;
    при выходе закрывает все окна OpenCV.

    Args:
        reader: источник кадров: из него берутся самый свежий кадр и время его получения.
        registry: реестр целей: из него берётся текущее состояние целей для рисования.

    Returns:
        None, когда пользователь закрыл окно или нажал q.
    """
    import cv2

    last_seq = 0
    try:
        while True:
            frame = reader.wait_for_new(last_seq, timeout=0.1) or reader.latest()
            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            last_seq = frame.seq
            # кадр читает и поток сопровождения, поэтому рисуем на копии
            canvas = frame.image.copy()
            draw_tracks(canvas, registry.views())
            if time.monotonic() - frame.timestamp > NO_SIGNAL_AFTER_SEC:
                cv2.putText(
                    canvas, "NO SIGNAL", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 2
                )
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # окно закрыли крестиком
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
