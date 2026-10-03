import pytest

from rectrack.overlay import annotated, draw_tracks, track_color
from rectrack.tracks import TrackView

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")  # рисование требует OpenCV; в окружении CI без него тесты пропускаются


def view(track_id=1, captured=True, box=(20.0, 30.0, 100.0, 120.0), trail=()):
    return TrackView(track_id, "person", 0.87, box, captured, list(trail))


def test_color_is_stable_and_differs_between_targets():
    assert track_color(3) == track_color(3)
    assert len({track_color(i) for i in range(1, 8)}) == 7
    assert all(0 <= channel <= 255 for channel in track_color(5))


def test_drawing_changes_the_pixels_inside_the_frame():
    canvas = np.zeros((200, 200, 3), np.uint8)
    draw_tracks(canvas, [view()])
    assert canvas.any()
    assert (canvas[30, 60] == track_color(1)).all()  # верхняя граница рамки


def test_captured_target_has_a_thicker_frame_than_a_pending_one():
    thick, thin = np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8)
    draw_tracks(thick, [view(captured=True)])
    draw_tracks(thin, [view(captured=False)])
    assert (thick.sum(axis=2) > 0).sum() > (thin.sum(axis=2) > 0).sum()


def test_trail_is_drawn():
    without, with_trail = np.zeros((200, 200, 3), np.uint8), np.zeros((200, 200, 3), np.uint8)
    draw_tracks(without, [view(box=(150, 150, 190, 190))])
    trail = [(10, 10), (60, 60), (120, 100)]
    draw_tracks(with_trail, [view(box=(150, 150, 190, 190), trail=trail)])
    assert with_trail[10:100, 10:100].any() and not without[10:100, 10:100].any()


def test_box_partly_outside_the_frame_is_fine():
    canvas = np.zeros((100, 100, 3), np.uint8)
    draw_tracks(canvas, [view(box=(-30.0, -30.0, 60.0, 60.0))])  # не должно бросить исключение


def test_annotated_returns_a_copy_and_leaves_the_original_alone():
    original = np.zeros((200, 200, 3), np.uint8)
    result = annotated(original, [view()])
    assert not original.any()
    assert result.any()
