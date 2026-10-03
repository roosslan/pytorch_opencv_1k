"""Фоновый поток: свежий кадр -> детектор с трекером -> реестр целей."""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from rectrack.capture import LatestFrameReader
from rectrack.detector import Observation
from rectrack.tracks import TrackRegistry

log = logging.getLogger(__name__)


class Tracker(Protocol):
    def track(self, frame_bgr: Any) -> list[Observation]:
        """Что обработка ждёт от детектора (реализация - YoloTracker.track): найти и сопроводить
        цели на одном кадре.

        Args:
            frame_bgr: кадр: массив numpy в порядке каналов BGR.

        Returns:
            список Observation для целей на этом кадре.
        """

    def reset(self) -> None:
        """Что обработка ждёт от детектора (реализация - YoloTracker.reset): забыть все цели,
        чтобы нумерация и сопоставление начались заново.
        """


class TrackingWorker:
    def __init__(
        self,
        reader: LatestFrameReader,
        detector: Tracker,
        registry: TrackRegistry,
        reset_gap: float = 5.0,
        stats_interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Создаёт фоновый обработчик кадров, но не запускает его (для запуска есть start()).

        Args:
            reader: источник кадров; обработчик всегда берёт самый свежий кадр.
            detector: детектор с трекером (YoloTracker или заглушка в тестах).
            registry: реестр целей, которому передаётся результат каждого кадра.
            reset_gap: если между обработанными кадрами прошло больше стольких секунд (поток
                пропадал), все цели закрываются с причиной 'gap', а трекер сбрасывается; по
                умолчанию 5 с.
            stats_interval: как часто (в секундах) писать в журнал статистику: кадры в секунду,
                время обработки кадра, число сопровождаемых целей; 0 - не писать.
            clock: монотонные часы для замера пауз и статистики; по умолчанию time.monotonic, в
                тестах подменяются.
        """
        self._reader = reader
        self._detector = detector
        self._registry = registry
        self._reset_gap = reset_gap
        self._stats_interval = stats_interval
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tracking", daemon=True)

    def start(self) -> None:
        """Запускает фоновый поток обработки кадров. Поток фоновый (daemon), поэтому не мешает
        завершению программы.
        """
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Просит поток обработки остановиться и ждёт его завершения. Кадр, который уже
        обрабатывается, будет обработан до конца.

        Args:
            timeout: сколько секунд ждать завершения потока; по умолчанию 10 с (с запасом на
                медленный прогон нейросети на процессоре).
        """
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        """Тело фонового потока обработки. В цикле:
          1. ждёт новый кадр (не дольше 0,5 с, чтобы вовремя заметить stop());
          2. если с прошлого обработанного кадра прошло больше reset_gap секунд, закрывает
             все цели с причиной 'gap' и сбрасывает трекер;
          3. прогоняет кадр через детектор; при ошибке пишет её в журнал, ждёт 1 с и берёт
             следующий кадр;
          4. передаёт результат в реестр целей с временем получения кадра, а не временем конца
             обработки, чтобы метки в базе не запаздывали на время работы нейросети;
          5. раз в stats_interval секунд пишет статистику в журнал.
        Работает, пока не вызван stop().
        """
        last_seq = 0
        last_done: float | None = None
        frames, busy, stats_since = 0, 0.0, self._clock()
        while not self._stop.is_set():
            frame = self._reader.wait_for_new(last_seq, timeout=0.5)
            if frame is None:
                continue
            last_seq = frame.seq

            if last_done is not None and self._clock() - last_done > self._reset_gap:
                # Поток пропадал: трекер считает кадры, а не время, и мог бы принять новых
                # людей за давно ушедших. Начинаем сопровождение с чистого листа.
                log.info("Разрыв в потоке больше %.0f с: сбрасываю трекер", self._reset_gap)
                self._registry.close_all("gap")
                self._detector.reset()

            started = self._clock()
            try:
                observations = self._detector.track(frame.image)
            except Exception:
                log.exception("Ошибка детектора")
                self._stop.wait(1.0)
                continue
            # время кадра, а не конца обработки: метки в базе не запаздывают на время нейросети
            self._registry.update(observations, frame.image, now=frame.timestamp)
            last_done = self._clock()

            frames += 1
            busy += last_done - started
            if self._stats_interval and last_done - stats_since >= self._stats_interval:
                span = last_done - stats_since
                log.info(
                    "Статистика: %.1f кадр/с, %.0f мс на кадр, целей в сопровождении: %d",
                    frames / span,
                    busy / frames * 1000,
                    self._registry.active_count(),
                )
                frames, busy, stats_since = 0, 0.0, last_done
