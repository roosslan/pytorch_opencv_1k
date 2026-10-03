"""Снимки в момент захвата цели."""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

_INVALID_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]')


def snapshot_stem(when: datetime, class_name: str, track_id: int) -> str:
    """Составляет имя файла снимка без расширения в виде дд-мм-гггг-чч-мм-сс-класс-idN, например
    21-09-2026-14-03-27-person-id5. Пробелы и символы, недопустимые в именах файлов Windows,
    в названии класса заменяются на «_».

    Args:
        when: момент снимка (обычно местное время захвата цели).
        class_name: название класса цели, например "person".
        track_id: номер цели от трекера.

    Returns:
        имя файла без расширения.
    """
    cls = _INVALID_IN_FILENAME.sub("_", class_name)
    return f"{when:%d-%m-%Y-%H-%M-%S}-{cls}-id{track_id}"


def jpeg_encode(image: Any, quality: int = 90) -> bytes:
    """Кодирует кадр в JPEG в памяти, без записи на диск.

    Args:
        image: изображение (массив numpy BGR).
        quality: качество JPEG от 0 до 100; по умолчанию 90 (хорошее качество при разумном размере
            файла).

    Returns:
        содержимое JPEG-файла в байтах.

    Raises:
        ValueError: OpenCV не смог закодировать изображение.
    """
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("не удалось закодировать кадр в JPEG")
    return buffer.tobytes()


class SnapshotWriter:
    def __init__(
        self,
        directory: str | Path,
        encode: Callable[[Any], bytes] = jpeg_encode,
        extension: str = ".jpg",
    ):
        """Настраивает сохранение снимков. Директория создаётся при первом сохранении, а не здесь.

        Args:
            directory: директория для снимков (строка или Path): абсолютная или относительно текущей
                директории программы.
            encode: функция, превращающая изображение в байты файла; по умолчанию JPEG с качеством
                90. В тестах подменяется, чтобы обойтись без OpenCV.
            extension: расширение файла вместе с точкой; по умолчанию ".jpg".
        """
        self._directory = Path(directory)
        self._encode = encode
        self._extension = extension

    def save(self, image: Any, when: datetime, class_name: str, track_id: int) -> Path:
        """Кодирует кадр и сохраняет его в директорию снимков под именем из snapshot_stem(). Сначала
        кодирует, потом создаёт файл, поэтому при ошибке кодирования пустой файл не остаётся.
        Существующие файлы никогда не перезаписываются: если имя занято (номер цели после
        сброса трекера мог повториться в ту же секунду), к нему добавляется « (2)», « (3)» и т. д.

        Args:
            image: кадр (массив numpy BGR), уже с нарисованными рамками, если они нужны.
            when: момент снимка для имени файла.
            class_name: название класса цели для имени файла.
            track_id: номер цели для имени файла.

        Returns:
            путь к сохранённому файлу.

        Raises:
            ValueError: кадр не удалось закодировать (для encode по умолчанию).
            OSError: не удалось создать директорию или записать файл.
        """
        data = self._encode(image)  # сначала кодируем: при ошибке не должно остаться пустого файла
        self._directory.mkdir(parents=True, exist_ok=True)
        stem = snapshot_stem(when, class_name, track_id)
        number = 1
        while True:
            suffix = "" if number == 1 else f" ({number})"
            path = self._directory / f"{stem}{suffix}{self._extension}"
            try:
                with path.open("xb") as fh:  # "x": никогда не перезаписывать
                    fh.write(data)
                return path
            except FileExistsError:
                number += 1
