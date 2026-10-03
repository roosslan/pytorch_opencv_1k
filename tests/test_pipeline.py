import threading

from rectrack.capture import Frame
from rectrack.detector import Observation
from rectrack.pipeline import TrackingWorker
from rectrack.tracks import TrackRegistry


def obs(track_id=1):
    return Observation(track_id, 0, "person", 0.9, (0.0, 0.0, 10.0, 10.0))


class FakeReader:
    """Выдаёт пронумерованные кадры, а когда новых нет, ждёт, как настоящий."""

    def __init__(self, images):
        self._images = list(images)
        self._idle = threading.Event()

    def wait_for_new(self, last_seq, timeout=1.0):
        if last_seq < len(self._images):
            return Frame(self._images[last_seq], last_seq + 1, 0.0)
        self._idle.wait(min(timeout, 0.05))
        return None


class FakeTracker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.frames = []
        self.resets = 0

    def track(self, frame):
        self.frames.append(frame)
        output = self.outputs.pop(0) if self.outputs else []
        if isinstance(output, Exception):
            raise output
        return output

    def reset(self):
        self.resets += 1


class Clock:
    """Каждый вызов даёт заданное время, чтобы можно было «пропустить» несколько секунд."""

    def __init__(self, times):
        self._times = list(times)
        self._last = times[-1]

    def __call__(self):
        if self._times:
            self._last = self._times.pop(0)
        return self._last


def wait_for(predicate, timeout=3.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        done.wait(0.01)
    return False


def run_worker(worker, condition):
    worker.start()
    try:
        assert wait_for(condition)
    finally:
        worker.stop()


def test_frames_go_through_the_detector_into_the_registry():
    events = []
    registry = TrackRegistry(
        on_captured=lambda t, f, v: events.append(("captured", f)), confirm_hits=2
    )
    detector = FakeTracker([[obs()], [obs()]])
    worker = TrackingWorker(FakeReader(["f1", "f2"]), detector, registry, stats_interval=0)
    run_worker(worker, lambda: events)
    assert detector.frames == ["f1", "f2"]
    assert events == [("captured", "f2")]  # снимок делается с кадра, на котором подтвердили


def test_detector_error_does_not_stop_the_worker(caplog):
    detector = FakeTracker([RuntimeError("модель упала"), [obs()]])
    registry = TrackRegistry(confirm_hits=1)
    worker = TrackingWorker(FakeReader(["bad", "good"]), detector, registry, stats_interval=0)
    # после ошибки поток спит секунду и берёт следующий кадр
    run_worker(worker, lambda: registry.active_count() == 1)
    assert "Ошибка детектора" in caplog.text


def test_a_long_gap_resets_the_tracker_and_closes_the_targets():
    lost = []
    registry = TrackRegistry(on_lost=lambda t, reason: lost.append(reason), confirm_hits=1)
    detector = FakeTracker([[obs()], [obs()]])
    # время спрашивается при старте, для каждого кадра дважды (до и после детектора) и ещё раз
    # при проверке разрыва; между кадрами проходит 60 секунд
    clock = Clock([0.0, 0.0, 0.0, 60.0, 60.0, 60.0, 60.0, 60.0])
    worker = TrackingWorker(
        FakeReader(["f1", "f2"]), detector, registry, reset_gap=5.0, stats_interval=0, clock=clock
    )
    run_worker(worker, lambda: detector.resets == 1)
    assert lost == ["gap"]


def test_short_pauses_do_not_reset_the_tracker():
    detector = FakeTracker([[obs()], [obs()], [obs()]])
    registry = TrackRegistry(confirm_hits=1)
    worker = TrackingWorker(
        FakeReader(["f1", "f2", "f3"]), detector, registry, reset_gap=5.0, stats_interval=0
    )
    run_worker(worker, lambda: len(detector.frames) == 3)
    assert detector.resets == 0


def test_statistics_are_logged(caplog):
    detector = FakeTracker([[obs()], [obs()]])
    registry = TrackRegistry(confirm_hits=1)
    clock = Clock([0.0] + [1.0] * 20)  # с самого начала «прошла секунда»
    worker = TrackingWorker(
        FakeReader(["f1", "f2"]), detector, registry, stats_interval=0.5, clock=clock
    )
    with caplog.at_level("INFO"):
        run_worker(worker, lambda: "Статистика" in caplog.text)
