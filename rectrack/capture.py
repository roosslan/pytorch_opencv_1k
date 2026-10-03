"""Захват RTSP-потока: всегда отдаёт самый свежий кадр.

Чтение и обработка идут в разных потоках. Поток чтения непрерывно вычитывает камеру, поэтому
буфер декодера не заполняется и задержка не копится, как бы медленно ни работала нейросеть.
Кадры, которые никто не успел забрать, отбрасываются.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

log = logging.getLogger(__name__)


class Frame(NamedTuple):
    image: Any
    seq: int  # растёт с каждым новым кадром, начинается с 1
    timestamp: float  # time.monotonic() в момент получения кадра


class LatestFrameReader:
    def __init__(
        self,
        open_capture: Callable[[], Any],
        reconnect_delay: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Создаёт читатель потока, но не запускает его (для запуска есть start()).

        Args:
            open_capture: функция без аргументов, которая открывает поток и возвращает объект с
                методами isOpened(), read() и release() (как cv2.VideoCapture). Вызывается при
                запуске и заново после каждой потери соединения.
            reconnect_delay: пауза в секундах перед повторным подключением после ошибки или потери
                потока; по умолчанию 3 с.
            clock: часы, которыми помечается момент получения каждого кадра; по умолчанию
                time.monotonic. В тестах подменяются управляемыми.
        """
        self._open_capture = open_capture
        self._reconnect_delay = reconnect_delay
        self._clock = clock
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

    def start(self) -> None:
        """Запускает фоновый поток чтения камеры. Поток фоновый (daemon), поэтому не мешает
        завершению программы.
        """
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Просит поток чтения остановиться, будит всех, кто ждёт кадр в wait_for_new(), и ждёт
        завершения потока.

        Args:
            timeout: сколько секунд ждать завершения потока; по умолчанию 5 с. Если чтение зависло
                дольше, метод всё равно вернёт управление: фоновый поток завершится вместе с
                программой.
        """
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout)

    def latest(self) -> Frame | None:
        """Возвращает последний полученный кадр сразу, не дожидаясь нового.

        Returns:
            Frame (изображение, порядковый номер, время получения) или None, если с момента запуска
            не пришло ни одного кадра.
        """
        with self._cond:
            return self._frame

    def wait_for_new(self, last_seq: int, timeout: float = 1.0) -> Frame | None:
        """Ждёт кадр новее того, который вызывающий уже обработал. Если такой кадр уже есть,
        возвращает его сразу. Промежуточные кадры не копятся: отдаётся только самый свежий.

        Args:
            last_seq: порядковый номер последнего обработанного кадра (0, если кадров ещё не было);
                нужен кадр с номером больше этого.
            timeout: сколько секунд ждать; по умолчанию 1 с.

        Returns:
            новый Frame или None, если за timeout новых кадров не пришло либо был вызван stop().
        """
        with self._cond:
            self._cond.wait_for(
                lambda: self._stop.is_set()
                or (self._frame is not None and self._frame.seq > last_seq),
                timeout,
            )
            frame = self._frame
            return frame if frame is not None and frame.seq > last_seq else None

    def _publish(self, image: Any) -> None:
        """Делает только что прочитанный кадр самым свежим: присваивает ему следующий порядковый
        номер, помечает временем получения и будит всех, кто ждёт в wait_for_new().
        Предыдущий кадр просто забывается.

        Args:
            image: изображение кадра (массив numpy в порядке каналов BGR, как отдаёт OpenCV).
        """
        with self._cond:
            seq = self._frame.seq + 1 if self._frame else 1
            self._frame = Frame(image, seq, self._clock())
            self._cond.notify_all()

    def _run(self) -> None:
        """Тело фонового потока чтения. В цикле открывает поток и читает кадры, пока они идут,
        публикуя каждый через _publish(). Если поток не открылся, оборвался или чтение
        завершилось ошибкой, пишет предупреждение в журнал, освобождает поток (release())
        и через reconnect_delay секунд подключается заново. Работает, пока не вызван stop().
        """
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._open_capture()
                if not capture.isOpened():
                    raise ConnectionError("не удалось открыть поток")
                log.info("Поток подключён")
                while not self._stop.is_set():
                    ok, image = capture.read()
                    if not ok or image is None:
                        log.warning("Поток потерян")
                        break
                    self._publish(image)
            except Exception as exc:
                log.warning("Ошибка захвата: %s", exc)
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        log.debug("release() завершился с ошибкой", exc_info=True)
            self._stop.wait(self._reconnect_delay)


def make_rtsp_capture(url: str, transport: str = "tcp", timeout_sec: float = 5.0):
    """Открывает RTSP-поток через FFmpeg-бэкенд OpenCV с настройками низкой задержки: без
    буферизации в FFmpeg (nobuffer, low_delay) и с буфером OpenCV на один кадр. Задаёт таймауты
    открытия и чтения, чтобы зависшая камера приводила к ошибке и переподключению, а не
    к вечному ожиданию. Настройки FFmpeg передаются через переменную окружения
    OPENCV_FFMPEG_CAPTURE_OPTIONS, поэтому она задаётся до открытия потока.

    Args:
        url: адрес потока, например rtsp://user:pass@192.168.1.10:554/1.
        transport: транспорт RTSP: "tcp" (надёжнее, по умолчанию) или "udp" (чуть меньше задержка,
            но возможны потери пакетов и артефакты на картинке).
        timeout_sec: таймаут открытия потока и чтения одного кадра в секундах; по умолчанию 5 с.

    Returns:
        cv2.VideoCapture. Открылся ли поток, вызывающий проверяет через isOpened().
    """
    import cv2

    # FFmpeg-бэкенд читает эту переменную при открытии потока, поэтому задаём её до него.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay"
    )
    timeout_ms = int(timeout_sec * 1000)
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
    )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture
