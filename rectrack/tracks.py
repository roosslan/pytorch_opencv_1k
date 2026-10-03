"""Жизненный цикл целей: захват -> сопровождение -> потеря.

Детектор с трекером (YOLOv8 + ByteTrack) на каждом кадре сообщает, какие цели он видит и под
какими номерами. Здесь эти сообщения превращаются в события:

* **обнаружена** цель, но ещё не подтверждена: молчим, чтобы случайные срабатывания
  детектора не попадали ни в базу, ни в снимки;
* **захват**: цель подтверждена на `confirm_hits` кадрах, о ней сообщается один раз;
* **сопровождение**: пока цель видна, копим статистику и раз в `points_interval` секунд
  отдаём точку траектории;
* **потеря**: цель не видна дольше `lost_timeout` секунд, о ней сообщается итог.

Модуль не зависит ни от torch, ни от OpenCV, ни от ClickHouse.
"""

import logging
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rectrack.detector import Observation

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def box_center(box: Box) -> Point:
    """Считает центр рамки.

    Args:
        box: рамка (x1, y1, x2, y2) в пикселях: левый верхний и правый нижний углы.

    Returns:
        точку (x, y) - центр рамки.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class Track:
    """Всё, что известно об одной цели."""

    uid: uuid.UUID  # уникален всегда, в отличие от track_id, который начинается заново при запуске
    track_id: int
    class_id: int
    class_name: str
    first_seen: float  # время (секунды с 1970 г., UTC) первого появления: для записи в базу
    last_seen: float
    hits: int  # на скольких кадрах цель была видна
    conf_sum: float
    conf_max: float
    first_box: Box
    last_box: Box
    trail: deque[Point]  # последние центры рамки, для рисования «хвоста»
    path_length: float = 0.0  # пройденный путь в пикселях
    captured_at: float | None = None
    last_point_at: float | None = None
    snapshot: str = ""  # путь к снимку в момент захвата
    streak: int = 0  # сколько кадров подряд цель видна (для подтверждения захвата)
    # Монотонное время (time.monotonic): по нему считаются таймауты и длительность, чтобы
    # перевод системных часов не ломал сопровождение. None, если трек создан не реестром.
    first_mono: float | None = None
    last_mono: float | None = None
    point_mono: float | None = None

    @property
    def captured(self) -> bool:
        """Захвачена ли цель, то есть подтверждена ли она на confirm_hits кадрах подряд.

        Returns:
            True, если момент захвата (captured_at) уже известен.
        """
        return self.captured_at is not None

    @property
    def conf_avg(self) -> float:
        """Средняя уверенность детектора по всем кадрам, на которых цель была видна.

        Returns:
            число от 0 до 1 (0, если кадров не было).
        """
        return self.conf_sum / self.hits if self.hits else 0.0

    @property
    def duration(self) -> float:
        """Сколько секунд цель находилась в кадре: от первого до последнего появления. Считается
        по монотонным часам, если они заданы (тогда перевод системных часов не искажает
        результат), иначе по системному времени.

        Returns:
            длительность в секундах.
        """
        if self.first_mono is not None and self.last_mono is not None:
            return self.last_mono - self.first_mono
        return self.last_seen - self.first_seen


@dataclass(frozen=True)
class TrackView:
    """Снимок состояния цели для рисования."""

    track_id: int
    class_name: str
    conf: float
    box: Box
    captured: bool
    trail: list[Point] = field(default_factory=list)


class TrackRegistry:
    def __init__(
        self,
        on_captured: Callable[[Track, Any, list[TrackView]], None] | None = None,
        on_point: Callable[[Track, Observation, float], None] | None = None,
        on_lost: Callable[[Track, str], None] | None = None,
        confirm_hits: int = 3,
        lost_timeout: float = 3.0,
        points_interval: float = 0.5,
        trail_length: int = 40,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        """Создаёт пустой реестр целей.

        Обработчики событий вызываются после снятия внутренней блокировки, чтобы медленный
        снимок (кодирование JPEG, запись на диск) не задерживал окно просмотра, которое читает
        цели из другого потока.

        Args:
            on_captured: вызывается, когда цель подтверждена: (цель, кадр, все цели на кадре для
                рисования снимка). None - не вызывать.
            on_point: вызывается для очередной точки траектории захваченной цели: (цель, наблюдение
                на кадре, время UTC в секундах). None - не вызывать.
            on_lost: вызывается, когда сопровождение захваченной цели закончено: (цель, причина
                'lost', 'gap' или 'shutdown'). None - не вызывать.
            confirm_hits: на скольких кадрах подряд цель должна быть видна, чтобы считаться
                захваченной; по умолчанию 3. Отсекает случайные срабатывания.
            lost_timeout: через сколько секунд невидимости цель считается потерянной; по умолчанию 3
                с.
            points_interval: как часто (в секундах) отдавать точку траектории; 0 - на каждом кадре.
                По умолчанию 0,5 с.
            trail_length: сколько последних центров рамки хранить для рисования «хвоста»; по
                умолчанию 40.
            clock: монотонные часы для таймаутов и длительностей; по умолчанию time.monotonic.
            wall_clock: системные часы для меток времени в базе; по умолчанию time.time.
        """
        self._on_captured = on_captured
        self._on_point = on_point
        self._on_lost = on_lost
        self._confirm_hits = confirm_hits
        self._lost_timeout = lost_timeout
        self._points_interval = points_interval
        self._trail_length = trail_length
        self._clock = clock
        self._wall_clock = wall_clock
        self._tracks: dict[int, Track] = {}
        self._lock = threading.Lock()  # окно читает цели из другого потока

    def update(
        self, observations: list[Observation], frame: Any = None, now: float | None = None
    ) -> None:
        """Учитывает результат детектора для одного кадра:
          - новую цель заводит, у известной обновляет рамку, траекторию и статистику;
          - цель, видимая confirm_hits кадров подряд, становится захваченной (on_captured);
          - для захваченных целей не чаще раза в points_interval отдаёт точку траектории
            (on_point);
          - цели, невидимые дольше lost_timeout, удаляет: о захваченных сообщает через on_lost,
            незахваченные исчезают молча как ложные срабатывания.
        Обработчики вызываются после снятия блокировки; их ошибки пишутся в журнал и не
        прерывают работу.

        Args:
            observations: цели на этом кадре от детектора с трекером.
            frame: сам кадр (массив numpy BGR); передаётся в on_captured для снимка.
            None: снимок сделать не получится.
            now: момент получения кадра по монотонным часам clock. None - текущий момент. Метка для
                базы получается пересчётом его в системное время.
        """
        pending: list[tuple[Callable | None, tuple]] = []
        with self._lock:
            current = self._clock()
            now = current if now is None else now
            wall = self._wall_clock() - (current - now)
            seen = set()
            for obs in observations:
                seen.add(obs.track_id)
                track = self._tracks.get(obs.track_id)
                if track is None:
                    track = self._start(obs, now, wall)
                    self._tracks[obs.track_id] = track
                else:
                    self._add(track, obs, now, wall)

                if not track.captured and track.streak >= self._confirm_hits:
                    track.captured_at = wall
                    views = self._views(now, observations)
                    pending.append((self._on_captured, (track, frame, views)))
                if track.captured and (
                    track.point_mono is None or now - track.point_mono >= self._points_interval
                ):
                    track.point_mono = now
                    track.last_point_at = wall
                    pending.append((self._on_point, (track, obs, wall)))

            for track_id, track in list(self._tracks.items()):
                if track_id not in seen:
                    track.streak = 0  # подтверждение требует кадров подряд
                if now - track.last_mono > self._lost_timeout:
                    del self._tracks[track_id]
                    # Цель, так и не дошедшая до захвата, исчезает молча: это была ложная тревога.
                    if track.captured:
                        pending.append((self._on_lost, (track, "lost")))
        for callback, args in pending:
            self._notify(callback, *args)

    def close_all(self, reason: str) -> None:
        """Завершает сопровождение всех целей сразу и очищает реестр. Для каждой захваченной цели
        вызывается on_lost с указанной причиной; незахваченные забываются молча.

        Args:
            reason: причина: 'gap' (разрыв видеопотока) или 'shutdown' (остановка программы).
        """
        with self._lock:
            tracks, self._tracks = list(self._tracks.values()), {}
        for track in tracks:
            if track.captured:
                self._notify(self._on_lost, track, reason)

    def views(self, max_age: float = 1.0) -> list[TrackView]:
        """Состояние целей для рисования в окне просмотра. Безопасно вызывать из другого потока.

        Args:
            max_age: цели, которых не было видно дольше стольких секунд, не рисуются (чтобы рамка не
                висела на месте ушедшего человека); по умолчанию 1 с.

        Returns:
            список TrackView (рамка, номер, класс, средняя уверенность, траектория, признак
            захвата).
        """
        with self._lock:
            return self._views(self._clock(), None, max_age)

    def active_count(self) -> int:
        """Сколько захваченных целей сейчас сопровождается (для статистики в журнале).

        Returns:
            число захваченных целей.
        """
        with self._lock:
            return sum(1 for track in self._tracks.values() if track.captured)

    def _start(self, obs: Observation, now: float, wall: float) -> Track:
        """Заводит новую цель по первому наблюдению: новый уникальный номер (uid), начальная
        статистика, траектория из одной точки, серия видимых подряд кадров равна 1.

        Args:
            obs: первое наблюдение цели.
            now: момент кадра по монотонным часам.
            wall: тот же момент в системном времени (секунды с 1970 г., UTC).

        Returns:
            новый Track.
        """
        center = box_center(obs.box)
        return Track(
            uid=uuid.uuid4(),
            track_id=obs.track_id,
            class_id=obs.class_id,
            class_name=obs.class_name,
            first_seen=wall,
            last_seen=wall,
            hits=1,
            conf_sum=obs.conf,
            conf_max=obs.conf,
            first_box=obs.box,
            last_box=obs.box,
            trail=deque([center], maxlen=self._trail_length),
            streak=1,
            first_mono=now,
            last_mono=now,
        )

    def _add(self, track: Track, obs: Observation, now: float, wall: float) -> None:
        """Добавляет к известной цели наблюдение с нового кадра: прибавляет к пройденному пути
        расстояние между центрами прошлой и новой рамки, продлевает траекторию, обновляет время
        последнего появления, рамку, число кадров, серию подряд и уверенность (сумму и максимум).

        Args:
            track: цель, которую нужно обновить (изменяется на месте).
            obs: наблюдение цели на новом кадре.
            now: момент кадра по монотонным часам.
            wall: тот же момент в системном времени (секунды с 1970 г., UTC).
        """
        previous = box_center(track.last_box)
        center = box_center(obs.box)
        track.path_length += math.dist(previous, center)
        track.trail.append(center)
        track.last_seen = wall
        track.last_mono = now
        track.streak += 1
        track.last_box = obs.box
        track.hits += 1
        track.conf_sum += obs.conf
        track.conf_max = max(track.conf_max, obs.conf)

    def _views(
        self, now: float, observations: list[Observation] | None, max_age: float = 1.0
    ) -> list[TrackView]:
        """Составляет список целей для рисования. Вызывать только под self._lock.
        Если переданы наблюдения текущего кадра, берутся только цели с этого кадра, с рамкой
        и уверенностью именно на нём (так рисуется снимок в момент захвата). Иначе берутся
        цели, видимые не дольше max_age секунд назад, с последней рамкой и средней уверенностью
        (так рисуется окно просмотра).

        Args:
            now: текущий момент по монотонным часам.
            observations: наблюдения текущего кадра или None.
            max_age: предел давности в секундах, если observations не заданы; по умолчанию 1 с.

        Returns:
            список TrackView.
        """
        current = {obs.track_id: obs for obs in observations} if observations is not None else None
        views = []
        for track_id, track in self._tracks.items():
            if current is not None:
                obs = current.get(track_id)
                if obs is None:
                    continue
                box, conf = obs.box, obs.conf
            elif now - track.last_mono <= max_age:
                box, conf = track.last_box, track.conf_sum / track.hits
            else:
                continue
            views.append(
                TrackView(track_id, track.class_name, conf, box, track.captured, list(track.trail))
            )
        return views

    @staticmethod
    def _notify(callback: Callable | None, *args: Any) -> None:
        """Безопасно вызывает обработчик события: если его нет, ничего не делает; если он упал,
        пишет ошибку в журнал, но не даёт ей прервать сопровождение.

        Args:
            callback: обработчик (on_captured, on_point или on_lost) или None.
            args: аргументы, с которыми его вызвать.
        """
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            log.exception("Ошибка в обработчике события цели")
