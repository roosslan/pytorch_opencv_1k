"""Детектор YOLOv8 (PyTorch) со встроенным трекером: находит цели и присваивает им номера."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from rectrack.config import ModelConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    """Одна цель на одном кадре."""

    track_id: int  # номер цели от трекера; уникален только в пределах одного запуска
    class_id: int
    class_name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях кадра


def resolve_classes(names: dict[int, str], wanted: list[str]) -> list[int] | None:
    """Переводит названия классов из конфигурации ([model] classes) в номера классов модели,
    которые понимает ultralytics.

    Args:
        names: словарь классов модели {номер: название}, например {0: "person", 2: "car"}.
        wanted: названия нужных классов из конфигурации; пустой список означает «все классы».

    Returns:
        список номеров в том же порядке, что и wanted, или None, если wanted пуст (для ultralytics
        None означает «без фильтра по классам»).

    Raises:
        ValueError: какого-то названия у модели нет; в тексте ошибки перечислены доступные
            классы.
    """
    if not wanted:
        return None
    by_name = {name: class_id for class_id, name in names.items()}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(
            f"У модели нет классов: {', '.join(unknown)}. "
            f"Доступные: {', '.join(sorted(by_name))}"
        )
    return [by_name[name] for name in wanted]


def _to_numpy(value: Any) -> np.ndarray:
    """Приводит тензор PyTorch (в том числе лежащий в памяти видеокарты) или любой массив
    к массиву numpy. Тензор сначала переносится в оперативную память (.cpu()).

    Args:
        value: тензор torch, массив numpy или последовательность чисел.

    Returns:
        np.ndarray с теми же данными.
    """
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def parse_result(result: Any, names: dict[int, str]) -> list[Observation]:
    """Разбирает результат ultralytics для одного кадра в список наблюдений Observation: рамка,
    уверенность, класс и номер цели от трекера для каждой найденной цели. Пока трекер не
    присвоил номер ни одной цели (boxes.id равен None: первые кадры или пустая сцена),
    возвращает пустой список.

    Args:
        result: объект Results из ultralytics для одного кадра (results[0] после model.track()).
        names: словарь классов модели {номер: название}; если номера класса в нём нет, названием
            станет сам номер в виде строки.

    Returns:
        список Observation, по одному на каждую цель с номером.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    conf = _to_numpy(boxes.conf)
    class_ids = _to_numpy(boxes.cls).astype(int)
    track_ids = _to_numpy(boxes.id).astype(int)
    return [
        Observation(
            track_id=int(track_id),
            class_id=int(class_id),
            class_name=names.get(int(class_id), str(int(class_id))),
            conf=float(score),
            box=(float(x1), float(y1), float(x2), float(y2)),
        )
        for (x1, y1, x2, y2), score, class_id, track_id in zip(
            xyxy, conf, class_ids, track_ids, strict=True
        )
    ]


def precision_kwargs(cfg: ModelConfig, supports_quantize: bool) -> dict[str, Any]:
    """Выбирает аргумент точности вычислений для вызова ultralytics. FP16 включается, только если
    он запрошен в конфигурации (half = true) и модель работает на видеокарте (device начинается
    с "cuda"): на процессоре FP16 не работает. Новые версии ultralytics принимают quantize=16
    и на каждый вызов со старым half пишут предупреждение; старые версии знают только half.

    Args:
        cfg: настройки модели; используются поля half и device.
        supports_quantize: True, если установленная версия ultralytics знает аргумент quantize.

    Returns:
        словарь для подстановки в model.track(): {"quantize": 16 или None} либо {"half": True или
        False}.
    """
    fp16 = cfg.half and cfg.device.startswith("cuda")
    if supports_quantize:
        return {"quantize": 16 if fp16 else None}
    return {"half": fp16}


class YoloTracker:
    """Обёртка над ultralytics: детекция и трекинг ByteTrack/BoT-SORT за один вызов.

    Тяжёлые библиотеки (torch, ultralytics) подключаются в load(), а не при импорте модуля."""

    def __init__(self, cfg: ModelConfig):
        """Запоминает настройки модели. Сама модель здесь не загружается (это долго и требует
        torch), для этого есть load().

        Args:
            cfg: настройки модели (раздел [model]): веса, устройство, размер входа, пороги, классы,
                точность, трекер.
        """
        self._cfg = cfg
        self._model: Any = None
        self._class_ids: list[int] | None = None
        self._precision: dict[str, Any] = {}

    def load(self) -> None:
        """Загружает модель YOLO. Импортирует torch и ultralytics; если в конфигурации указана
        видеокарта, проверяет, что CUDA доступна; загружает веса (если файла нет, ultralytics
        скачает их сам); переводит названия нужных классов в номера; выбирает аргумент
        точности. Отключает отправку анонимной статистики ultralytics (sync=False).

        Raises:
            RuntimeError: device = "cuda", а CUDA недоступна.
            ValueError: у модели нет нужного класса.
            Exception: ошибки ultralytics при загрузке или скачивании весов.
        """
        import torch
        import ultralytics
        from ultralytics import YOLO

        device = self._cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"В конфигурации device = '{device}', но CUDA недоступна: скорее всего, "
                "установлен PyTorch без поддержки CUDA (см. README)"
            )
        if ultralytics.settings.get("sync"):
            # ultralytics по умолчанию отправляет анонимную статистику: выключаем навсегда
            # (настройка сохраняется в файле настроек ultralytics пользователя)
            ultralytics.settings.update({"sync": False})
            log.info("Отправка статистики ultralytics отключена (sync=False)")
        self._model = YOLO(self._cfg.weights)  # если файла нет, веса будут скачаны
        self._class_ids = resolve_classes(self._model.names, self._cfg.classes)
        from ultralytics.cfg import DEFAULT_CFG_DICT

        self._precision = precision_kwargs(self._cfg, "quantize" in DEFAULT_CFG_DICT)
        log.info(
            "Модель %s, устройство %s, классы: %s",
            self._cfg.weights,
            device,
            ", ".join(self._cfg.classes) or "все",
        )

    def track(self, frame_bgr: Any) -> list[Observation]:
        """Прогоняет один кадр через детектор и трекер. Трекер помнит цели между вызовами
        (persist=True), поэтому одна и та же цель на соседних кадрах получает один номер.
        Перед первым вызовом должен быть выполнен load().

        Args:
            frame_bgr: кадр: массив numpy формы (высота, ширина, 3) в порядке каналов BGR, как
                отдаёт OpenCV.

        Returns:
            список Observation для целей, которым трекер присвоил номера.
        """
        cfg = self._cfg
        results = self._model.track(
            frame_bgr,
            persist=True,  # трекер помнит цели между вызовами
            verbose=False,
            tracker=cfg.tracker,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            classes=self._class_ids,
            device=cfg.device,
            **self._precision,
        )
        return parse_result(results[0], self._model.names)

    def reset(self) -> None:
        """Сбрасывает трекер: все цели забываются, на следующем кадре нумерация начнётся заново.
        Нужен после разрыва потока: трекер считает кадры, а не секунды, и без сброса мог бы
        принять нового человека за того, кто ушёл несколько минут назад. В ultralytics трекеры
        живут внутри предиктора, поэтому сбрасывается предиктор целиком; при следующем
        track() он создаётся заново. До load() ничего не делает.
        """
        if self._model is not None:
            self._model.predictor = None  # трекеры создаются вместе с предиктором
