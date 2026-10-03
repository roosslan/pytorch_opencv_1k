"""Загрузка и проверка конфигурации (TOML)."""

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Пароль ClickHouse лучше не хранить в config.toml: его можно задать переменной окружения
# или строкой в файле .env рядом с config.toml. Порядок: окружение, .env, config.toml.
PASSWORD_ENV = "REC_CLICKHOUSE_PASSWORD"
DOTENV_NAME = ".env"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TYPE_NAMES = {
    int: "целым числом",
    float: "числом",
    str: "строкой",
    bool: "true/false",
    list: "списком",
}


class ConfigError(Exception):
    """Файл конфигурации не найден или содержит ошибку."""


@dataclass(frozen=True)
class CameraConfig:
    name: str = "cam1"  # имя камеры: попадает во все таблицы ClickHouse
    rtsp_url: str = ""
    transport: str = "tcp"  # "tcp" надёжнее, "udp" даёт чуть меньшую задержку
    reconnect_delay_sec: float = 3.0
    timeout_sec: float = 5.0  # таймаут открытия/чтения, зависший поток вызовет переподключение


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "models/yolov8n.pt"  # если файла нет, ultralytics скачает его сам
    device: str = "cpu"  # "cpu", "cuda", "cuda:0"
    imgsz: int = 640  # размер, до которого сжимается кадр; больше = видны мелкие цели, но медленнее
    conf: float = 0.35  # порог уверенности детектора
    iou: float = 0.5  # порог NMS
    classes: list[str] = field(default_factory=lambda: ["person"])  # пустой список = все классы
    half: bool = False  # FP16, имеет смысл только на GPU
    tracker: str = "bytetrack.yaml"  # "bytetrack.yaml", "botsort.yaml" или путь к своему файлу


@dataclass(frozen=True)
class TrackingConfig:
    confirm_hits: int = 3  # цель считается захваченной после стольких подряд подтверждённых кадров
    lost_timeout_sec: float = 3.0  # цель потеряна, если её не видно дольше этого времени
    points_interval_sec: float = 0.5  # как часто писать точку траектории (0 = на каждом кадре)
    trail_length: int = 40  # длина «хвоста» траектории в окне
    reset_gap_sec: float = 5.0  # разрыв между кадрами больше этого сбрасывает трекер
    stats_interval_sec: float = 30.0  # как часто писать в журнал статистику (0 = не писать)


@dataclass(frozen=True)
class SnapshotsConfig:
    enabled: bool = True
    directory: str = "capture"
    annotate: bool = True  # рисовать рамки и номера целей на снимке


@dataclass(frozen=True)
class ClickHouseConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8123  # HTTP-порт
    username: str = "default"
    password: str = ""
    database: str = "rec"
    secure: bool = False  # HTTPS
    create_schema: bool = True  # при старте создать базу и таблицы, если их нет
    batch_size: int = 500  # писать, когда накопилось столько строк
    flush_interval_sec: float = 2.0  # ...или когда прошло столько секунд
    max_buffer_rows: int = 100_000  # если ClickHouse недоступен, копим не больше стольких строк
    connect_timeout_sec: float = 5.0


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)


def redact_url(url: str) -> str:
    """Прячет логин и пароль в адресе, чтобы адрес можно было безопасно писать в журнал:
    rtsp://admin:secret@10.0.0.5/1 -> rtsp://***@10.0.0.5/1. Адрес без учётных данных
    возвращается без изменений.

    Args:
        url: адрес, обычно RTSP-адрес камеры.

    Returns:
        адрес, в котором всё до «@» в части с хостом заменено на «***».
    """
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def load_config(path: str | Path) -> Config:
    """Читает файл конфигурации TOML, проверяет его (см. parse_config) и подставляет пароль
    ClickHouse из внешних источников. Пароль берётся из первого источника, где он задан:
    переменная окружения REC_CLICKHOUSE_PASSWORD, затем строка с тем же именем в файле .env
    рядом с файлом конфигурации, затем значение password из самого config.toml.

    Args:
        path: путь к файлу конфигурации (строка или Path).

    Returns:
        готовый объект Config.

    Raises:
        ConfigError: файла нет, в нём ошибка синтаксиса TOML, неизвестный раздел или ключ,
            значение неверного типа или вне допустимого диапазона, либо .env не читается.
    """
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(
            f"Файл конфигурации '{path}' не найден. "
            f"Скопируйте config.example.toml в {path} и отредактируйте."
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Некорректный TOML в '{path}': {exc}") from exc
    config = parse_config(raw)
    password = os.environ.get(PASSWORD_ENV) or read_dotenv(path.parent / DOTENV_NAME).get(
        PASSWORD_ENV
    )
    if password:
        config = dataclasses.replace(
            config, clickhouse=dataclasses.replace(config.clickhouse, password=password)
        )
    return config


def read_dotenv(path: Path) -> dict[str, str]:
    """Читает простой файл .env со строками вида КЛЮЧ=значение. Пустые строки, комментарии (#...)
    и строки без «=» пропускаются. Пробелы вокруг ключа и значения обрезаются, одна пара
    одинаковых кавычек вокруг значения ("..." или '...') снимается. Метка BOM в начале файла
    (её добавляет Блокнот Windows) не мешает. Подстановка переменных и многострочные значения
    не поддерживаются.

    Args:
        path: путь к файлу .env.

    Returns:
        словарь {ключ: значение}; пустой, если файла нет.

    Raises:
        ConfigError: файл есть, но прочитать его не удалось.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать '{path}': {exc}") from exc
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_config(raw: dict[str, Any]) -> Config:
    """Превращает уже разобранный TOML (словарь) в объект Config. Каждый раздел ([camera],
    [model], [tracking], [snapshots], [clickhouse]) собирается в свой класс настроек;
    отсутствующие разделы и ключи получают значения по умолчанию. Затем все значения
    проверяются на допустимость (см. _validate).

    Args:
        raw: словарь из tomllib.load(): {имя раздела: {ключ: значение}}.

    Returns:
        проверенный Config.

    Raises:
        ConfigError: неизвестный раздел или ключ, неверный тип или недопустимое значение.
    """
    sections = {f.name: f for f in dataclasses.fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Неизвестные разделы конфигурации: {', '.join(sorted(unknown))}")

    built = {
        name: _build_section(sections[name].default_factory, name, raw.get(name, {}))
        for name in sections
    }
    config = Config(**built)
    _validate(config)
    return config


def _build_section(cls: type, name: str, data: Any) -> Any:
    """Собирает один раздел конфигурации в объект класса настроек и проверяет типы значений.
    Ожидаемый тип каждого ключа берётся из его значения по умолчанию в классе. Целое число
    принимается там, где ожидается дробное (conf = 1 вместо 1.0), но true/false вместо числа
    не принимается. У списков дополнительно проверяется, что все элементы - строки.

    Args:
        cls: класс настроек раздела (CameraConfig, ModelConfig и т. д.).
        name: имя раздела, как в TOML, например "camera"; нужно только для текста ошибок.
        data: содержимое раздела из TOML: {ключ: значение}.

    Returns:
        экземпляр cls; ключи, которых нет в data, получают значения по умолчанию.

    Raises:
        ConfigError: раздел не является таблицей, в нём есть неизвестный ключ или значение
            неверного типа.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] должен быть таблицей")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Неизвестные ключи в [{name}]: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        expected = type(getattr(cls(), key))
        # целое число подходит там, где ждут дробное (например, `conf = 1`)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(
                f"[{name}] {key} должен быть {_TYPE_NAMES[expected]}, "
                f"а сейчас {type(value).__name__}"
            )
        if expected is list and not all(isinstance(item, str) for item in value):
            raise ConfigError(f"[{name}] {key} должен быть списком строк")
    return cls(**data)


def _validate(config: Config) -> None:
    """Проверяет, что значения всей конфигурации допустимы:
      - обязательные строки не пустые (имя камеры, адрес потока, путь к весам и т. п.);
      - числа в разумных пределах (порог уверенности в (0, 1), порт 1..65535, таймауты
        положительные и т. д.);
      - транспорт только "tcp" или "udp";
      - имя базы - допустимый идентификатор ClickHouse (оно подставляется прямо в SQL);
      - лимит буфера не меньше размера пакета.
    Останавливается на первой найденной ошибке.

    Args:
        config: собранная конфигурация, которую нужно проверить.

    Raises:
        ConfigError: найдено недопустимое значение; в тексте описана первая ошибка.
    """
    cam, mdl, trk = config.camera, config.model, config.tracking
    snap, ch = config.snapshots, config.clickhouse
    _require(bool(cam.name), "[camera] name не должен быть пустым")
    _require(bool(cam.rtsp_url), "[camera] rtsp_url обязателен")
    _require(cam.transport in ("tcp", "udp"), "[camera] transport должен быть 'tcp' или 'udp'")
    _require(cam.reconnect_delay_sec >= 0, "[camera] reconnect_delay_sec должен быть >= 0")
    _require(cam.timeout_sec > 0, "[camera] timeout_sec должен быть > 0")

    _require(bool(mdl.weights), "[model] weights не должен быть пустым")
    _require(bool(mdl.device), "[model] device не должен быть пустым")
    _require(32 <= mdl.imgsz <= 4096, "[model] imgsz должен быть в диапазоне 32..4096")
    _require(0 < mdl.conf < 1, "[model] conf должен быть в диапазоне (0, 1)")
    _require(0 < mdl.iou <= 1, "[model] iou должен быть в диапазоне (0, 1]")
    _require(bool(mdl.tracker), "[model] tracker не должен быть пустым")

    _require(trk.confirm_hits >= 1, "[tracking] confirm_hits должен быть >= 1")
    _require(trk.lost_timeout_sec > 0, "[tracking] lost_timeout_sec должен быть > 0")
    _require(trk.points_interval_sec >= 0, "[tracking] points_interval_sec должен быть >= 0")
    _require(trk.trail_length >= 1, "[tracking] trail_length должен быть >= 1")
    _require(trk.reset_gap_sec > 0, "[tracking] reset_gap_sec должен быть > 0")
    _require(trk.stats_interval_sec >= 0, "[tracking] stats_interval_sec должен быть >= 0")

    _require(bool(snap.directory), "[snapshots] directory не должен быть пустым")

    _require(bool(ch.host), "[clickhouse] host не должен быть пустым")
    _require(1 <= ch.port <= 65535, "[clickhouse] port должен быть в диапазоне 1..65535")
    _require(
        bool(_IDENTIFIER.match(ch.database)),
        "[clickhouse] database может состоять только из латинских букв, цифр и '_'",
    )
    _require(ch.batch_size >= 1, "[clickhouse] batch_size должен быть >= 1")
    _require(ch.flush_interval_sec > 0, "[clickhouse] flush_interval_sec должен быть > 0")
    _require(
        ch.max_buffer_rows >= ch.batch_size,
        "[clickhouse] max_buffer_rows не может быть меньше batch_size",
    )
    _require(ch.connect_timeout_sec > 0, "[clickhouse] connect_timeout_sec должен быть > 0")


def _require(condition: bool, message: str) -> None:
    """Вспомогательная проверка для _validate: если условие ложно, выбрасывает ConfigError.

    Args:
        condition: условие, которое должно выполняться.
        message: текст ошибки для пользователя: что не так и в каком разделе.

    Raises:
        ConfigError: условие ложно; текст ошибки - message.
    """
    if not condition:
        raise ConfigError(message)
