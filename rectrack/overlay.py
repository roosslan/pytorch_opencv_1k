"""Рисование целей на кадре: рамки, номера, «хвосты» траекторий."""

import colorsys
from typing import Any

from rectrack.tracks import TrackView


def track_color(track_id: int) -> tuple[int, int, int]:
    """Даёт устойчивый цвет для номера цели: одна и та же цель всегда рисуется одним цветом,
    а соседние номера получают заметно разные цвета (оттенок сдвигается на золотое сечение).

    Args:
        track_id: номер цели от трекера.

    Returns:
        цвет (синий, зелёный, красный), каждая составляющая 0..255, как ждёт OpenCV.
    """
    hue = (track_id * 0.61803398875) % 1.0  # золотое сечение: соседние номера сильно различаются
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(canvas: Any, views: list[TrackView]) -> None:
    """Рисует цели прямо на переданном изображении (изменяет его): рамку, линию траектории
    («хвост» из последних центров рамки) и подпись «класс #номер уверенность» на цветной
    плашке над рамкой. Захваченные цели обводятся толстой рамкой, ещё не подтверждённые -
    тонкой.

    Args:
        canvas: изображение (массив numpy BGR), на котором рисовать; изменяется на месте.
        views: состояния целей: рамка, номер, класс, уверенность, траектория и признак захвата.
    """
    import cv2

    for view in views:
        color = track_color(view.track_id)
        x1, y1, x2, y2 = (int(v) for v in view.box)
        thickness = 3 if view.captured else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if len(view.trail) > 1:
            points = [(int(x), int(y)) for x, y in view.trail]
            for start, end in zip(points, points[1:], strict=False):
                cv2.line(canvas, start, end, color, 2)

        label = f"{view.class_name} #{view.track_id} {view.conf:.2f}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        top = max(y1 - height - 8, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + width + 6, top + height + 8), color, cv2.FILLED)
        cv2.putText(
            canvas, label, (x1 + 3, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )


def annotated(frame: Any, views: list[TrackView]) -> Any:
    """Возвращает копию кадра с нарисованными целями; исходный кадр не меняется. Используется
    для снимков в момент захвата.

    Args:
        frame: исходный кадр (массив numpy BGR).
        views: цели для рисования (см. draw_tracks).

    Returns:
        новый массив numpy с рамками, траекториями и подписями.
    """
    canvas = frame.copy()
    draw_tracks(canvas, views)
    return canvas
