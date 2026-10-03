import numpy as np
import pytest

from rectrack.config import ModelConfig
from rectrack.detector import Observation, parse_result, precision_kwargs, resolve_classes

NAMES = {0: "person", 1: "bicycle", 2: "car"}


class FakeTensor:
    """Похож на тензор PyTorch: .cpu().numpy(), как у результатов на GPU."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids):
        self.xyxy, self.conf, self.cls, self.id = xyxy, conf, cls, ids


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def test_resolve_classes_maps_names_to_ids():
    assert resolve_classes(NAMES, ["car", "person"]) == [2, 0]


def test_empty_class_list_means_all_classes():
    assert resolve_classes(NAMES, []) is None


def test_unknown_class_lists_the_available_ones():
    with pytest.raises(ValueError, match="humen") as error:
        resolve_classes(NAMES, ["person", "humen"])
    assert "bicycle" in str(error.value)


def test_parse_result_with_numpy_arrays():
    boxes = FakeBoxes(
        xyxy=np.array([[10, 20, 110, 220], [0, 0, 5, 5]], dtype=np.float32),
        conf=np.array([0.9, 0.5], dtype=np.float32),
        cls=np.array([0, 2], dtype=np.float32),
        ids=np.array([7, 8], dtype=np.float32),
    )
    result = parse_result(FakeResult(boxes), NAMES)
    assert result[0] == Observation(7, 0, "person", pytest.approx(0.9), (10.0, 20.0, 110.0, 220.0))
    assert (result[1].track_id, result[1].class_name) == (8, "car")


def test_parse_result_with_torch_like_tensors():
    boxes = FakeBoxes(
        xyxy=FakeTensor([[1, 2, 3, 4]]),
        conf=FakeTensor([0.75]),
        cls=FakeTensor([1]),
        ids=FakeTensor([3]),
    )
    [obs] = parse_result(FakeResult(boxes), NAMES)
    assert (obs.track_id, obs.class_name, obs.box) == (3, "bicycle", (1.0, 2.0, 3.0, 4.0))


def test_no_tracks_yet_gives_empty_list():
    """Пока трекер ничего не ведёт, ultralytics отдаёт boxes.id = None."""
    boxes = FakeBoxes(np.zeros((0, 4)), np.zeros(0), np.zeros(0), None)
    assert parse_result(FakeResult(boxes), NAMES) == []


def test_result_without_boxes_gives_empty_list():
    assert parse_result(FakeResult(None), NAMES) == []


def test_unknown_class_id_falls_back_to_its_number():
    boxes = FakeBoxes(
        np.array([[0, 0, 1, 1]]), np.array([0.5]), np.array([42]), np.array([1])
    )
    assert parse_result(FakeResult(boxes), NAMES)[0].class_name == "42"


def test_precision_uses_quantize_on_new_ultralytics():
    """Новые версии ultralytics предупреждают о `half` на каждом кадре: передаём `quantize`."""
    gpu = ModelConfig(device="cuda", half=True)
    assert precision_kwargs(gpu, supports_quantize=True) == {"quantize": 16}
    fp32 = ModelConfig(device="cuda")
    assert precision_kwargs(fp32, supports_quantize=True) == {"quantize": None}


def test_fp16_is_never_requested_on_cpu():
    cpu = ModelConfig(device="cpu", half=True)
    assert precision_kwargs(cpu, supports_quantize=True) == {"quantize": None}
    assert precision_kwargs(cpu, supports_quantize=False) == {"half": False}


def test_precision_falls_back_to_half_on_old_ultralytics():
    gpu = ModelConfig(device="cuda:0", half=True)
    assert precision_kwargs(gpu, supports_quantize=False) == {"half": True}
