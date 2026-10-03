import uuid
from collections import deque
from datetime import datetime

from rectrack.detector import Observation
from rectrack.recorder import TargetRecorder
from rectrack.storage import COLUMNS, EVENTS, POINTS, TRACKS
from rectrack.tracks import Track, TrackView

SESSION = uuid.UUID(int=1)
WHEN = datetime(2026, 9, 21, 14, 3, 27)


class FakeSink:
    def __init__(self):
        self.rows = []

    def add(self, table, row):
        self.rows.append((table, row))


class FakeSnapshots:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save(self, image, when, class_name, track_id):
        if self.fail:
            raise OSError("диск полон")
        self.saved.append((image, when, class_name, track_id))
        return f"capture/{class_name}-{track_id}.jpg"


def make_track(captured=True):
    return Track(
        uid=uuid.UUID(int=2),
        track_id=5,
        class_id=0,
        class_name="person",
        first_seen=1000.0,
        last_seen=1004.0,
        hits=10,
        conf_sum=8.0,
        conf_max=0.9,
        first_box=(0.0, 0.0, 1.0, 1.0),
        last_box=(1.0, 1.0, 2.0, 2.0),
        trail=deque(),
        captured_at=1001.0 if captured else None,
    )


def views():
    return [TrackView(5, "person", 0.9, (0.0, 0.0, 1.0, 1.0), True)]


def named(table, row):
    return dict(zip(COLUMNS[table], row, strict=True))


def test_capture_saves_an_annotated_snapshot_and_writes_the_event():
    sink, snapshots = FakeSink(), FakeSnapshots()
    recorder = TargetRecorder(
        "cam1",
        SESSION,
        sink,
        snapshots,
        annotate=lambda frame, v: f"{frame}+рамки",
        now=lambda: WHEN,
    )
    track = make_track()
    recorder.on_captured(track, "кадр", views())

    assert snapshots.saved == [("кадр+рамки", WHEN, "person", 5)]
    assert track.snapshot == "capture/person-5.jpg"
    [(table, row)] = sink.rows
    event = named(EVENTS, row)
    assert table == EVENTS
    assert (event["event"], event["track_id"], event["camera"]) == ("captured", 5, "cam1")
    assert event["snapshot"] == "capture/person-5.jpg"


def test_snapshot_without_annotation_is_the_plain_frame():
    snapshots = FakeSnapshots()
    recorder = TargetRecorder("cam1", SESSION, None, snapshots, annotate=None, now=lambda: WHEN)
    recorder.on_captured(make_track(), "кадр", views())
    assert snapshots.saved[0][0] == "кадр"


def test_failed_snapshot_does_not_stop_the_event_from_being_written(caplog):
    sink = FakeSink()
    recorder = TargetRecorder("cam1", SESSION, sink, FakeSnapshots(fail=True))
    track = make_track()
    recorder.on_captured(track, "кадр", views())
    assert track.snapshot == ""
    assert len(sink.rows) == 1
    assert "Не удалось сохранить снимок" in caplog.text


def test_no_snapshot_without_a_frame():
    snapshots = FakeSnapshots()
    recorder = TargetRecorder("cam1", SESSION, None, snapshots)
    recorder.on_captured(make_track(), None, views())
    assert snapshots.saved == []


def test_point_goes_to_the_points_table():
    sink = FakeSink()
    recorder = TargetRecorder("cam1", SESSION, sink)
    recorder.on_point(make_track(), Observation(5, 0, "person", 0.8, (1.0, 2.0, 3.0, 4.0)), 1002.0)
    [(table, row)] = sink.rows
    point = named(POINTS, row)
    assert table == POINTS
    assert (point["x1"], point["y2"], point["conf"]) == (1.0, 4.0, 0.8)


def test_loss_writes_an_event_and_the_summary():
    sink = FakeSink()
    recorder = TargetRecorder("cam1", SESSION, sink)
    recorder.on_lost(make_track(), "gap")
    assert [table for table, _ in sink.rows] == [EVENTS, TRACKS]
    assert named(EVENTS, sink.rows[0][1])["event"] == "lost"
    assert named(TRACKS, sink.rows[1][1])["close_reason"] == "gap"


def test_everything_works_without_a_database_and_snapshots(caplog):
    recorder = TargetRecorder("cam1", SESSION)
    track = make_track()
    with caplog.at_level("INFO"):
        recorder.on_captured(track, "кадр", views())
        recorder.on_point(track, Observation(5, 0, "person", 0.8, (0, 0, 1, 1)), 1.0)
        recorder.on_lost(track, "lost")
    assert "Захват" in caplog.text and "Потеря" in caplog.text


def test_snapshot_name_uses_the_capture_time_by_default():
    snapshots = FakeSnapshots()
    recorder = TargetRecorder("cam1", SESSION, None, snapshots, annotate=None)
    recorder.on_captured(make_track(), "кадр", views())
    [(_, when, _, _)] = snapshots.saved
    assert when == datetime.fromtimestamp(1001.0)  # captured_at, в местном времени
