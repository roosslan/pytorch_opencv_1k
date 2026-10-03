"""Запись в ClickHouse: буфер, пакетная вставка, переживание недоступной базы."""

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rectrack.config import ClickHouseConfig
from rectrack.detector import Observation
from rectrack.tracks import Track

log = logging.getLogger(__name__)

EVENTS = "events"
TRACKS = "tracks"
POINTS = "track_points"

# Порядок колонок должен совпадать с rectrack/schema.sql (это проверяет тест).
COLUMNS: dict[str, list[str]] = {
    EVENTS: [
        "ts", "camera", "event", "session_id", "track_uid", "track_id",
        "class_name", "conf", "duration_sec", "snapshot",
    ],
    TRACKS: [
        "camera", "session_id", "track_uid", "track_id", "class_name", "captured_at",
        "last_seen", "duration_sec", "hits", "conf_max", "conf_avg", "path_length_px",
        "first_box", "last_box", "snapshot", "close_reason",
    ],
    POINTS: ["ts", "camera", "track_uid", "track_id", "x1", "y1", "x2", "y2", "conf"],
}  # fmt: skip

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# Если буфер переполнен, в первую очередь выбрасываем то, что дешевле всего потерять.
DROP_ORDER = (POINTS, TRACKS, EVENTS)

ClientFactory = Callable[[ClickHouseConfig], Any]


def utc(timestamp: float) -> datetime:
    """Переводит время в секундах с 1970 г. в datetime с явным часовым поясом UTC, как ждёт
    клиент ClickHouse для колонок DateTime64(3, 'UTC').

    Args:
        timestamp: время в секундах с 1970 г. (как возвращает time.time()).

    Returns:
        datetime в UTC.
    """
    return datetime.fromtimestamp(timestamp, tz=UTC)


def event_row(
    camera: str, session_id: uuid.UUID, track: Track, event: str, ts: float
) -> list[Any]:
    """Составляет строку для таблицы events в порядке колонок COLUMNS[EVENTS].

    Args:
        camera: имя камеры.
        session_id: уникальный номер текущего запуска программы.
        track: цель, к которой относится событие; из неё берутся уникальный номер, номер от трекера,
            класс, максимальная уверенность, длительность и путь к снимку.
        event: тип события: 'captured' (захват) или 'lost' (потеря). Длительность записывается
            только для 'lost', для остальных событий пишется 0.
        ts: время события в секундах с 1970 г. (UTC).

    Returns:
        список значений колонок.
    """
    duration = track.duration if event == "lost" else 0.0
    return [
        utc(ts), camera, event, session_id, track.uid, track.track_id,
        track.class_name, track.conf_max, duration, track.snapshot,
    ]  # fmt: skip


def track_row(camera: str, session_id: uuid.UUID, track: Track, reason: str) -> list[Any]:
    """Составляет итоговую строку по цели для таблицы tracks в порядке колонок COLUMNS[TRACKS]:
    момент захвата и последнего появления, длительность, число кадров, максимальная и средняя
    уверенность, пройденный путь, первая и последняя рамки, снимок, причина закрытия.

    Args:
        camera: имя камеры.
        session_id: уникальный номер текущего запуска программы.
        track: завершённая цель со всей статистикой. Если момента захвата нет, вместо него пишется
            момент первого появления.
        reason: причина закрытия: 'lost', 'gap' или 'shutdown'.

    Returns:
        список значений колонок.
    """
    captured_at = track.captured_at if track.captured_at is not None else track.first_seen
    return [
        camera, session_id, track.uid, track.track_id, track.class_name,
        utc(captured_at), utc(track.last_seen), track.duration, track.hits,
        track.conf_max, track.conf_avg, track.path_length,
        list(track.first_box), list(track.last_box), track.snapshot, reason,
    ]  # fmt: skip


def point_row(camera: str, track: Track, obs: Observation, ts: float) -> list[Any]:
    """Составляет строку точки траектории для таблицы track_points в порядке колонок
    COLUMNS[POINTS]: время, камера, цель, рамка (x1, y1, x2, y2) и уверенность.

    Args:
        camera: имя камеры.
        track: цель, к которой относится точка (нужны её уникальный номер и номер от трекера).
        obs: наблюдение цели на кадре: рамка в пикселях и уверенность.
        ts: время кадра в секундах с 1970 г. (UTC).

    Returns:
        список значений колонок.
    """
    x1, y1, x2, y2 = obs.box
    return [utc(ts), camera, track.uid, track.track_id, x1, y1, x2, y2, obs.conf]


def schema_statements(database: str) -> list[str]:
    """Читает rectrack/schema.sql, подставляет имя базы вместо {database}, убирает строки,
    целиком состоящие из комментария (--), и разбивает текст по «;» на отдельные запросы:
    ClickHouse выполняет за один вызов только один запрос.

    Args:
        database: имя базы ClickHouse. Подставляется прямо в SQL, поэтому должно быть заранее
            проверено (это делает проверка конфигурации).

    Returns:
        список SQL-запросов без завершающих «;».
    """
    text = SCHEMA_FILE.read_text(encoding="utf-8").replace("{database}", database)
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def default_client_factory(cfg: ClickHouseConfig) -> Any:
    """Создаёт клиент clickhouse-connect (HTTP-протокол) по настройкам: адрес, порт, учётные
    данные, HTTPS, таймауты. Серверная сессия не используется, чтобы запросы не мешали друг
    другу и не зависели от времени её жизни. Библиотека импортируется здесь, а не при загрузке
    модуля, чтобы тестам она была не нужна.

    Args:
        cfg: настройки ClickHouse (раздел [clickhouse]).

    Returns:
        подключённый клиент.

    Raises:
        Exception: ошибки clickhouse-connect - сервер недоступен или отверг учётные данные.
    """
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        secure=cfg.secure,
        connect_timeout=cfg.connect_timeout_sec,
        send_receive_timeout=send_receive_timeout(cfg),
        # без серверной сессии: запросы не мешают друг другу и не зависят от её времени жизни
        autogenerate_session_id=False,
    )


def send_receive_timeout(cfg: ClickHouseConfig) -> float:
    """Считает таймаут одного запроса к ClickHouse (отправка данных и ожидание ответа): вдвое
    больше таймаута подключения, но не меньше 30 секунд, чтобы крупная пачка строк успевала
    записаться.

    Args:
        cfg: настройки ClickHouse; используется connect_timeout_sec.

    Returns:
        таймаут в секундах.
    """
    return max(cfg.connect_timeout_sec * 2, 30)


def ensure_schema(client: Any, database: str) -> None:
    """Создаёт базу и таблицы по schema.sql, если их ещё нет (запросы CREATE ... IF NOT EXISTS),
    поэтому вызывать можно сколько угодно раз. Уже существующие таблицы не изменяются.

    Args:
        client: клиент ClickHouse.
        database: имя базы.

    Raises:
        Exception: ошибки ClickHouse при выполнении запросов.
    """
    for statement in schema_statements(database):
        client.command(statement)


def recent_events(client: Any, database: str, count: int) -> list[Any]:
    """Читает последние события из таблицы events, новые сверху.

    Args:
        client: клиент ClickHouse.
        database: имя базы.
        count: сколько событий вернуть; передаётся в запрос как параметр, а не вставкой в текст
            запроса.

    Returns:
        список кортежей (время, событие, камера, класс, номер цели, длительность, путь к снимку).
    """
    result = client.query(
        "SELECT ts, event, camera, class_name, track_id, duration_sec, snapshot "
        f"FROM {database}.events ORDER BY ts DESC LIMIT {{n:UInt32}}",
        parameters={"n": count},
    )
    return list(result.result_rows)


class ClickHouseSink:
    """Копит строки в памяти и пишет их пачками в фоновом потоке.

    Если ClickHouse недоступен, строки остаются в буфере и отправляются, когда база вернётся.
    Буфер ограничен `max_buffer_rows`: при переполнении выбрасываются самые старые точки
    траекторий, затем итоги целей, затем события. Из-за недоступной базы сопровождение
    никогда не тормозит и не падает."""

    def __init__(
        self,
        cfg: ClickHouseConfig,
        client_factory: ClientFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Создаёт буфер записи в ClickHouse. Ни подключения, ни фонового потока здесь ещё нет:
        поток запускает start(), а подключение происходит при первой записи.

        Args:
            cfg: настройки ClickHouse: адрес, учётные данные, размер пакета, интервал записи, лимит
                буфера, создавать ли схему.
            client_factory: функция (настройки) -> клиент ClickHouse; None - настоящий клиент
                clickhouse-connect (default_client_factory). В тестах подменяется.
            clock: монотонные часы для пауз между повторами и для частоты предупреждений о
                переполнении; по умолчанию time.monotonic.
        """
        self._cfg = cfg
        # выбирается в момент вызова, чтобы default_client_factory можно было подменить
        self._client_factory = client_factory or default_client_factory
        self._clock = clock
        self._lock = threading.Lock()  # буферы
        # Запись идёт по одной: stop() может вызвать flush(), пока фоновый поток ещё пишет.
        self._flush_lock = threading.Lock()
        self._buffers: dict[str, deque[list[Any]]] = {table: deque() for table in COLUMNS}
        self._client: Any = None
        self._schema_ready = not cfg.create_schema
        self._failures = 0  # неудачных попыток подряд
        self._last_attempt = float("-inf")
        self._dropped = 0
        self._last_drop_log = float("-inf")
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="clickhouse-writer", daemon=True)

    def start(self) -> None:
        """Запускает фоновый поток записи в ClickHouse. Поток фоновый (daemon), поэтому не мешает
        завершению программы.
        """
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Останавливает фоновый поток записи и пытается дописать всё, что осталось в буфере.
        Если база недоступна, пишет в журнал, сколько строк так и не записано.

        Args:
            timeout: сколько секунд ждать завершения фонового потока; None (по умолчанию) - таймаут
                одного запроса плюс таймаут подключения, чтобы уже начатая запись успела
                закончиться.
        """
        self._stop.set()
        self._wake.set()
        if timeout is None:
            timeout = send_receive_timeout(self._cfg) + self._cfg.connect_timeout_sec
        if self._thread.is_alive():
            self._thread.join(timeout)
        if not self.flush():
            log.error("В ClickHouse не записаны %d строк: база недоступна", self.pending())

    def add(self, table: str, row: list[Any]) -> None:
        """Кладёт строку в буфер таблицы. Сама запись в базу происходит позже в фоновом потоке:
        по таймеру (flush_interval_sec) или сразу, как только в буфере набралось batch_size
        строк. Если буфер превысил max_buffer_rows, выбрасываются самые старые строки: сначала
        точки траекторий, потом итоги целей, потом события. Метод никогда не ждёт базу, поэтому
        не тормозит сопровождение.

        Args:
            table: имя таблицы: EVENTS, TRACKS или POINTS.
            row: значения колонок в порядке COLUMNS[table] (см. event_row, track_row, point_row).
        """
        with self._lock:
            self._buffers[table].append(row)
            self._trim()
            full = self._pending() >= self._cfg.batch_size
        if full:
            self._wake.set()

    def pending(self) -> int:
        """Сколько строк сейчас ждёт записи во всех таблицах вместе.

        Returns:
            число строк в буфере.
        """
        with self._lock:
            return self._pending()

    def flush(self) -> bool:
        """Записывает в базу всё накопленное прямо сейчас. Одновременно идёт только одна запись:
        если фоновый поток уже пишет, метод дождётся его. Незаписанное остаётся в буфере.

        Returns:
            True, если записано всё (или записывать было нечего); False, если база недоступна и
            строки остались в буфере.
        """
        with self._flush_lock:
            return self._flush()

    def _flush(self) -> bool:
        """Сама запись, без защиты от одновременной записи (её обеспечивает flush()). Забирает все
        строки из буферов, подключается (при необходимости создавая схему) и вставляет строки
        пачками, по одной вставке на таблицу. Если вставка упала, таблицы, записанные до этого,
        считаются записанными, а остальные строки возвращаются в начало буфера; клиент
        сбрасывается, чтобы в следующий раз подключиться заново. Подробная ошибка пишется
        в журнал только при первой неудаче подряд, восстановление связи - один раз.

        Returns:
            True при успехе (или если записывать было нечего), False при ошибке.
        """
        with self._lock:
            remaining = {table: list(rows) for table, rows in self._buffers.items() if rows}
            for buffer in self._buffers.values():
                buffer.clear()
        if not remaining:
            return True

        try:
            client = self._connect()
            for table in list(remaining):
                client.insert(
                    table,
                    remaining[table],
                    column_names=COLUMNS[table],
                    database=self._cfg.database,
                )
                del remaining[table]  # эта таблица записана
        except Exception as exc:
            self._client = None  # в следующий раз подключимся заново
            self._failures += 1
            if self._failures == 1:  # подробно только первая неудача подряд
                log.error("Не удалось записать в ClickHouse, буду повторять: %s", exc)
            self._requeue(remaining)
            return False

        if self._failures:
            log.info("Связь с ClickHouse восстановлена")
        self._failures = 0
        return True

    # Внутреннее -------------------------------------------------------------------------

    def _pending(self) -> int:
        """То же, что pending(), но без захвата блокировки: вызывать только под self._lock.

        Returns:
            число строк во всех буферах.
        """
        return sum(len(buffer) for buffer in self._buffers.values())

    def _connect(self) -> Any:
        """Возвращает клиент ClickHouse, создавая его при первом вызове или после ошибки. Если
        в настройках create_schema = true, после первого успешного подключения создаёт базу
        и таблицы.

        Returns:
            клиент ClickHouse.

        Raises:
            Exception: ошибки подключения или создания схемы.
        """
        if self._client is None:
            self._client = self._client_factory(self._cfg)
        if not self._schema_ready:
            ensure_schema(self._client, self._cfg.database)
            self._schema_ready = True
            log.info("Схема ClickHouse готова (база '%s')", self._cfg.database)
        return self._client

    def _requeue(self, unsent: dict[str, list[list[Any]]]) -> None:
        """Возвращает незаписанные строки в начало буферов в исходном порядке, перед строками,
        которые пришли во время записи. Затем обрезает буфер до лимита (см. _trim).

        Args:
            unsent: {имя таблицы: список строк}, которые не удалось записать.
        """
        with self._lock:
            for table, rows in unsent.items():
                self._buffers[table].extendleft(reversed(rows))
            self._trim()

    def _trim(self) -> None:
        """Не даёт буферу вырасти больше max_buffer_rows: выбрасывает самые старые строки, сначала
        точки траекторий, потом итоги целей и только потом события. Предупреждение о выброшенных
        строках пишется в журнал не чаще раза в 30 секунд. Вызывать только под self._lock.
        """
        excess = self._pending() - self._cfg.max_buffer_rows
        if excess <= 0:
            return
        for table in DROP_ORDER:
            buffer = self._buffers[table]
            while excess > 0 and buffer:
                buffer.popleft()
                excess -= 1
                self._dropped += 1
        now = self._clock()
        if now - self._last_drop_log >= 30:  # не чаще раза в 30 секунд
            self._last_drop_log = now
            log.warning("Буфер ClickHouse переполнен, выброшено строк: %d", self._dropped)

    def _run(self) -> None:
        """Тело фонового потока записи. Просыпается раз в flush_interval_sec или раньше, когда
        add() набрал полный пакет, и записывает буфер через flush(). После неудач делает паузы
        между попытками всё длиннее: интервал * 2^(число неудач подряд), но не больше 30 с,
        чтобы не долбить недоступную базу. Работает, пока не вызван stop().
        """
        interval = self._cfg.flush_interval_sec
        while not self._stop.is_set():
            self._wake.wait(interval)
            self._wake.clear()
            if self._stop.is_set():
                break
            # После неудачи ждём всё дольше (до 30 с), а не долбим недоступную базу.
            backoff = min(interval * 2**self._failures, 30.0) if self._failures else 0.0
            if self._clock() - self._last_attempt < backoff:
                continue
            self._last_attempt = self._clock()
            self.flush()
