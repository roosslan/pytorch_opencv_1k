import threading

from rectrack.capture import LatestFrameReader


class ScriptedCapture:
    """Отдаёт заданные кадры, потом либо сообщает о потере потока, либо «висит» до release()."""

    def __init__(self, frames, opened=True, then_fail=False):
        self._frames = list(frames)
        self._opened = opened
        self._then_fail = then_fail
        self.released = threading.Event()

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        if self._then_fail:
            return False, None
        self.released.wait(0.2)  # живой поток, в котором пока нет новых данных
        return False, None

    def release(self):
        self.released.set()


def wait_until(reader, seq):
    """Дождаться кадра с указанным порядковым номером."""
    frame = reader.latest()
    while frame is None or frame.seq < seq:
        frame = reader.wait_for_new(frame.seq if frame else 0, timeout=2)
        assert frame is not None, f"кадр {seq} так и не пришёл"
    return frame


def test_latest_is_none_before_any_frame():
    assert LatestFrameReader(lambda: ScriptedCapture([])).latest() is None


def test_reader_keeps_only_the_newest_frame():
    capture = ScriptedCapture(["f1", "f2", "f3"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        frame = wait_until(reader, 3)
        assert (frame.image, frame.seq) == ("f3", 3)
        assert reader.latest() == frame
    finally:
        reader.stop()


def test_wait_for_new_returns_none_when_nothing_newer_arrives():
    capture = ScriptedCapture(["f1"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        frame = wait_until(reader, 1)
        assert reader.wait_for_new(frame.seq, timeout=0.05) is None
    finally:
        reader.stop()


def test_wait_for_new_returns_immediately_if_a_newer_frame_exists():
    capture = ScriptedCapture(["f1", "f2"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        wait_until(reader, 2)
        frame = reader.wait_for_new(1, timeout=0.05)
        assert frame is not None and frame.image == "f2"
    finally:
        reader.stop()


def test_reader_reconnects_after_the_stream_is_lost():
    captures = [ScriptedCapture(["a1"], then_fail=True), ScriptedCapture(["b1", "b2"])]
    opened = []

    def factory():
        capture = captures[len(opened)]
        opened.append(capture)
        return capture

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        assert wait_until(reader, 3).image == "b2"
        assert len(opened) == 2
        assert captures[0].released.is_set()  # сломанное соединение закрыто
    finally:
        reader.stop()


def test_reader_retries_when_the_stream_cannot_be_opened():
    captures = [ScriptedCapture([], opened=False), ScriptedCapture(["ok"])]
    opened = []

    def factory():
        capture = captures[len(opened)]
        opened.append(capture)
        return capture

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        assert wait_until(reader, 1).image == "ok"
    finally:
        reader.stop()


def test_reader_survives_an_exception_from_the_factory():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("сбой")
        return ScriptedCapture(["ok"])

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        assert wait_until(reader, 1).image == "ok"
    finally:
        reader.stop()


def test_stop_ends_the_thread():
    capture = ScriptedCapture([])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    reader.stop()
    assert not reader._thread.is_alive()
