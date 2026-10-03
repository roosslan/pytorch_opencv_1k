import logging

import pytest

from rectrack.detector import Observation
from rectrack.tracks import TrackRegistry, box_center


def obs(track_id, box=(0.0, 0.0, 10.0, 10.0), conf=0.9, name="person"):
    return Observation(track_id, 0, name, conf, box)


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def make(**kwargs):
    """Реестр, который записывает все события в список."""
    events = []
    # одни и те же часы для монотонного и системного времени: метки в событиях равны `now`
    kwargs.setdefault("clock", Clock(0.0))
    kwargs.setdefault("wall_clock", kwargs["clock"])
    registry = TrackRegistry(
        on_captured=lambda track, frame, views: events.append(("captured", track.track_id, views)),
        on_point=lambda track, o, ts: events.append(("point", track.track_id, ts)),
        on_lost=lambda track, reason: events.append(("lost", track.track_id, reason)),
        **kwargs,
    )
    return registry, events


def kinds(events):
    return [event[0] for event in events]


def test_box_center():
    assert box_center((0, 0, 10, 20)) == (5, 10)


def test_target_is_captured_after_confirm_hits_and_only_once():
    registry, events = make(confirm_hits=3)
    registry.update([obs(1)], now=0.0)
    registry.update([obs(1)], now=0.25)
    assert events == []  # ещё не подтверждена: молчим
    registry.update([obs(1)], now=0.5)
    assert kinds(events) == ["captured", "point"]
    registry.update([obs(1)], now=0.75)
    assert kinds(events).count("captured") == 1


def test_capture_gets_the_frame_and_views_of_the_current_targets():
    registry, events = make(confirm_hits=2)
    registry.update([obs(1), obs(2)], frame="кадр-1", now=0.0)
    registry.update([obs(1)], frame="кадр-2", now=0.25)  # цель 1 подтверждена, цели 2 нет в кадре
    [captured] = [event for event in events if event[0] == "captured"]
    views = captured[2]
    assert [(v.track_id, v.captured) for v in views] == [(1, True)]


def test_unconfirmed_target_disappears_silently():
    """Одиночное ложное срабатывание не должно попасть ни в события, ни в базу."""
    registry, events = make(confirm_hits=3, lost_timeout=3.0)
    registry.update([obs(1)], now=0.0)
    registry.update([], now=4.0)
    assert events == []
    assert registry.active_count() == 0


def test_captured_target_is_reported_lost_after_the_timeout():
    registry, events = make(confirm_hits=1, lost_timeout=3.0)
    registry.update([obs(1)], now=0.0)
    registry.update([], now=2.9)
    assert "lost" not in kinds(events)
    registry.update([], now=3.1)
    assert ("lost", 1, "lost") in events
    registry.update([], now=10.0)
    assert kinds(events).count("lost") == 1  # второй раз не сообщается


def test_a_brief_gap_keeps_the_same_target():
    """Закрыли на секунду и снова появилась: тот же номер, тот же захват, никакой потери."""
    registry, events = make(confirm_hits=1, lost_timeout=3.0)
    registry.update([obs(1)], now=0.0)
    registry.update([], now=1.0)
    registry.update([obs(1)], now=2.0)
    assert kinds(events).count("captured") == 1
    assert "lost" not in kinds(events)


def test_same_id_after_a_loss_is_a_new_target():
    registry, events = make(confirm_hits=1, lost_timeout=1.0)
    registry.update([obs(1)], now=0.0)
    registry.update([], now=2.0)  # потеряна
    registry.update([obs(1)], now=5.0)  # тот же номер от трекера, но уже другая цель
    assert kinds(events).count("captured") == 2
    assert kinds(events).count("lost") == 1


def test_target_uids_are_unique_even_for_the_same_track_id():
    uids = []
    registry = TrackRegistry(
        on_captured=lambda track, frame, views: uids.append(track.uid),
        confirm_hits=1,
        lost_timeout=1.0,
    )
    registry.update([obs(1)], now=0.0)
    registry.update([], now=2.0)
    registry.update([obs(1)], now=5.0)
    assert len(uids) == 2 and uids[0] != uids[1]


def test_points_follow_the_interval():
    registry, events = make(confirm_hits=3, points_interval=0.5)
    for step in range(9):  # 0.0 ... 2.0 через 0.25 с
        registry.update([obs(1)], now=step * 0.25)
    points = [event[2] for event in events if event[0] == "point"]
    assert points == [0.5, 1.0, 1.5, 2.0]  # первая точка в момент захвата, дальше раз в 0.5 с


def test_zero_interval_gives_a_point_on_every_frame_after_capture():
    registry, events = make(confirm_hits=1, points_interval=0.0)
    for step in range(4):
        registry.update([obs(1)], now=float(step))
    assert kinds(events).count("point") == 4


def test_no_points_before_capture():
    registry, events = make(confirm_hits=5)
    for step in range(4):
        registry.update([obs(1)], now=float(step))
    assert events == []


def test_track_statistics():
    tracks = []
    registry = TrackRegistry(
        on_lost=lambda track, reason: tracks.append(track), confirm_hits=1, lost_timeout=1.0
    )
    registry.update([obs(1, box=(0, 0, 10, 10), conf=0.5)], now=0.0)
    registry.update([obs(1, box=(3, 4, 13, 14), conf=0.9)], now=1.0)  # центр сместился на (3, 4)
    registry.update([obs(1, box=(3, 4, 13, 14), conf=0.7)], now=2.0)
    registry.update([], now=4.0)
    [track] = tracks
    assert track.hits == 3
    assert track.path_length == pytest.approx(5.0)
    assert track.conf_max == pytest.approx(0.9)
    assert track.conf_avg == pytest.approx(0.7)
    assert track.duration == pytest.approx(2.0)
    assert track.first_box == (0, 0, 10, 10)
    assert track.last_box == (3, 4, 13, 14)


def test_close_all_reports_captured_targets_only():
    registry, events = make(confirm_hits=2)
    registry.update([obs(1), obs(2)], now=0.0)
    registry.update([obs(1)], now=0.25)  # 1 захвачена, 2 нет
    registry.close_all("shutdown")
    assert ("lost", 1, "shutdown") in events
    assert not any(event[:2] == ("lost", 2) for event in events)
    assert registry.active_count() == 0


def test_close_all_twice_does_not_repeat():
    registry, events = make(confirm_hits=1)
    registry.update([obs(1)], now=0.0)
    registry.close_all("gap")
    registry.close_all("shutdown")
    assert kinds(events).count("lost") == 1


def test_views_show_recent_targets_with_their_trail():
    clock = Clock(100.0)
    registry = TrackRegistry(confirm_hits=1, clock=clock)
    registry.update([obs(1, box=(0, 0, 10, 10))], now=100.0)
    registry.update([obs(1, box=(10, 0, 20, 10))], now=100.2)
    [view] = registry.views(max_age=1.0)
    assert (view.track_id, view.captured) == (1, True)
    assert view.trail == [(5.0, 5.0), (15.0, 5.0)]
    clock.now = 101.5
    assert registry.views(max_age=1.0) == []  # давно не видели: не рисуем


def test_trail_length_is_limited():
    clock = Clock(0.0)
    registry = TrackRegistry(confirm_hits=1, trail_length=3, clock=clock)
    for step in range(10):
        registry.update([obs(1, box=(step, 0, step + 10, 10))], now=float(step) * 0.1)
    clock.now = 1.0
    [view] = registry.views()
    assert len(view.trail) == 3
    assert view.trail[-1] == (14.0, 5.0)


def test_active_count_counts_only_captured_targets():
    registry, _ = make(confirm_hits=2)
    registry.update([obs(1), obs(2)], now=0.0)
    registry.update([obs(1)], now=0.25)
    assert registry.active_count() == 1


def test_error_in_a_handler_does_not_break_tracking(caplog):
    calls = []

    def broken(track, frame, views):
        raise RuntimeError("не получилось")

    clock = Clock(0.0)
    registry = TrackRegistry(
        on_captured=broken,
        on_point=lambda t, o, ts: calls.append(ts),
        confirm_hits=1,
        clock=clock,
        wall_clock=clock,
    )
    with caplog.at_level(logging.ERROR):
        registry.update([obs(1)], now=0.0)
    assert calls == [0.0]  # точка всё равно записана
    assert "Ошибка в обработчике" in caplog.text


def test_time_comes_from_the_clock_when_not_given():
    clock = Clock(500.0)
    registry, events = make(confirm_hits=1, clock=clock)
    registry.update([obs(1)])
    assert ("point", 1, 500.0) in events


def test_confirmation_requires_consecutive_frames():
    """Цель, которую видно через кадр, не подтверждается: нужны кадры подряд."""
    registry, events = make(confirm_hits=3, lost_timeout=3.0)
    for step in range(6):
        registry.update([obs(1)] if step % 2 == 0 else [], now=step * 0.25)
    assert events == []
    for step in range(6, 9):  # теперь три кадра подряд
        registry.update([obs(1)], now=step * 0.25)
    assert "captured" in kinds(events)


def test_wall_clock_jump_does_not_affect_timeouts_or_duration():
    """Системные часы перевели на час назад: цель не теряется и длительность не портится."""
    mono, wall = Clock(0.0), Clock(1_000_000.0)
    tracks = []
    registry = TrackRegistry(
        on_lost=lambda track, reason: tracks.append(track),
        confirm_hits=1,
        lost_timeout=3.0,
        clock=mono,
        wall_clock=wall,
    )
    registry.update([obs(1)], now=0.0)
    mono.now, wall.now = 1.0, 1_000_001.0 - 3600
    registry.update([obs(1)], now=1.0)
    assert registry.active_count() == 1
    mono.now = 5.0
    registry.update([], now=5.0)
    [track] = tracks
    assert track.duration == pytest.approx(1.0)
    assert track.first_seen == pytest.approx(1_000_000.0)


def test_timestamps_are_taken_at_frame_time():
    """Метка для базы соответствует моменту получения кадра, а не концу обработки."""
    mono, wall = Clock(10.5), Clock(2000.5)  # обработка закончилась через 0.5 с после кадра
    points = []
    registry = TrackRegistry(
        on_point=lambda t, o, ts: points.append(ts), confirm_hits=1, clock=mono, wall_clock=wall
    )
    registry.update([obs(1)], now=10.0)
    assert points == [pytest.approx(2000.0)]


def test_handlers_run_without_holding_the_lock():
    """Пока сохраняется снимок, окно может читать цели (иначе была бы взаимоблокировка)."""
    seen = []
    registry = None

    def on_captured(track, frame, views):
        seen.append(len(registry.views(max_age=10.0)))  # берёт ту же блокировку

    clock = Clock(0.0)
    registry = TrackRegistry(on_captured=on_captured, confirm_hits=1, clock=clock)
    registry.update([obs(1)], now=0.0)
    assert seen == [1]
