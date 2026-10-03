from datetime import datetime

import pytest

from rectrack.snapshots import SnapshotWriter, jpeg_encode, snapshot_stem

WHEN = datetime(2026, 9, 21, 14, 3, 27)


def test_stem_format():
    assert snapshot_stem(WHEN, "person", 5) == "21-09-2026-14-03-27-person-id5"


def test_stem_replaces_characters_that_are_invalid_in_file_names():
    stem = snapshot_stem(WHEN, "traffic light/red", 1)
    assert stem == "21-09-2026-14-03-27-traffic_light_red-id1"


def test_writer_creates_the_folder_and_writes_the_encoded_image(tmp_path):
    writer = SnapshotWriter(tmp_path / "a" / "b", encode=lambda image: b"jpeg")
    path = writer.save("кадр", WHEN, "person", 5)
    assert path == tmp_path / "a" / "b" / "21-09-2026-14-03-27-person-id5.jpg"
    assert path.read_bytes() == b"jpeg"


def test_writer_never_overwrites_an_existing_file(tmp_path):
    writer = SnapshotWriter(tmp_path, encode=lambda image: image)
    first = writer.save(b"1", WHEN, "person", 5)
    second = writer.save(b"2", WHEN, "person", 5)
    third = writer.save(b"3", WHEN, "person", 5)
    assert [p.name for p in (first, second, third)] == [
        "21-09-2026-14-03-27-person-id5.jpg",
        "21-09-2026-14-03-27-person-id5 (2).jpg",
        "21-09-2026-14-03-27-person-id5 (3).jpg",
    ]
    assert first.read_bytes() == b"1"


def test_encoding_failure_leaves_no_file_and_no_folder(tmp_path):
    def broken(_):
        raise ValueError("не закодировалось")

    writer = SnapshotWriter(tmp_path / "capture", encode=broken)
    with pytest.raises(ValueError):
        writer.save("кадр", WHEN, "person", 5)
    assert not (tmp_path / "capture").exists()


def test_jpeg_encode_produces_a_jpeg():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    data = jpeg_encode(np.zeros((32, 32, 3), np.uint8))
    assert data[:2] == b"\xff\xd8"  # начало любого JPEG
