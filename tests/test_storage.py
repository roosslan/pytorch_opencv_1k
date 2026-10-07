import re
import time
import uuid
from collections import deque
from datetime import UTC, datetime

import pytest

from rectrack import storage
from rectrack.config import ClickHouseConfig
from rectrack.detector import Observation
from rectrack.storage import (
    COLUMNS,
    EVENTS,
    POINTS,
    TRACKS,
    ClickHouseSink,
    ensure_schema,
    event_row,
    point_row,
    recent_events,
    retry_delay,
    schema_statements,
    track_row,
)
from rectrack.tracks import Track

CAMERA = "cam1"
SESSION = uuid.UUID(int=1)


def make_track():
    return Track(
        uid=uuid.UUID(int=2),
        track_id=7,
        class_id=0,
        class_name="person",
        first_seen=1000.0,
        last_seen=1010.5,
        hits=20,
        conf_sum=16.0,
        conf_max=0.95,
        first_box=(1.0, 2.0, 3.0, 4.0),
        last_box=(5.0, 6.0, 7.0, 8.0),
        trail=deque(),
        path_length=123.0,
        captured_at=1001.0,
        snapshot="capture/x.jpg",
    )


class FakeClient:
    def __init__(self):
        self.commands = []
        self.inserts = []  # (таблица, строки, колонки, база)
        self.fail_inserts = False
        self.queries = []

    def command(self, sql):
        self.commands.append(sql)

    def insert(self, table, rows, column_names=None, database=None):
        if self.fail_inserts:
            raise ConnectionError("ClickHouse не отвечает")
        self.inserts.append((table, list(rows), column_names, database))

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        return type("Result", (), {"result_rows": [("строка",)]})()


def make_sink(client=None, **overrides):
    client = client or FakeClient()
    params = {"batch_size": 3, "max_buffer_rows": 10, "create_schema": False}
    params.update(overrides)
    cfg = ClickHouseConfig(**params)
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return client

    return ClickHouseSink(cfg, client_factory=factory), client, factory_calls


# --- схема и строки -------------------------------------------------------------------


def schema_columns():
    """Колонки каждой таблицы из schema.sql в порядке объявления."""
    text = storage.SCHEMA_FILE.read_text(encoding="utf-8")
    tables = {}
    pattern = r"CREATE TABLE IF NOT EXISTS \{database\}\.(\w+)\s*\((.*?)\)\s*ENGINE"
    for match in re.finditer(pattern, text, re.S):
        names = []
        for line in match.group(2).splitlines():
            line = line.split("--")[0].strip()
            if line:
                names.append(line.split()[0])
        tables[match.group(1)] = names
    return tables


def test_columns_match_the_schema_in_the_same_order():
    assert schema_columns() == COLUMNS


def test_schema_is_split_into_single_statements():
    statements = schema_statements("cams")
    assert len(statements) == 4
    assert statements[0] == "CREATE DATABASE IF NOT EXISTS cams"
    assert all("{database}" not in s and not s.startswith("--") for s in statements)
    assert all("cams." in s for s in statements[1:])


def test_ensure_schema_runs_every_statement():
    client = FakeClient()
    ensure_schema(client, "rec")
    assert len(client.commands) == 4


@pytest.mark.parametrize(
    "table, row",
    [
        (EVENTS, lambda t: event_row(CAMERA, SESSION, t, "captured", 1001.0)),
        (EVENTS, lambda t: event_row(CAMERA, SESSION, t, "lost", 1010.5)),
        (TRACKS, lambda t: track_row(CAMERA, SESSION, t, "lost")),
        (POINTS, lambda t: point_row(CAMERA, t, Observation(7, 0, "person", 0.9, (1, 2, 3, 4)), 1)),
    ],
)
def test_row_has_one_value_per_column(table, row):
    assert len(row(make_track())) == len(COLUMNS[table])


def test_event_rows_carry_duration_only_for_lost():
    track = make_track()
    captured = event_row(CAMERA, SESSION, track, "captured", 1001.0)
    lost = event_row(CAMERA, SESSION, track, "lost", 1010.5)
    duration = COLUMNS[EVENTS].index("duration_sec")
    assert captured[duration] == 0.0
    assert lost[duration] == pytest.approx(10.5)
    assert captured[COLUMNS[EVENTS].index("snapshot")] == "capture/x.jpg"


def test_times_are_utc_datetimes():
    row = track_row(CAMERA, SESSION, make_track(), "lost")
    captured_at = row[COLUMNS[TRACKS].index("captured_at")]
    assert captured_at == datetime.fromtimestamp(1001.0, tz=UTC)
    assert captured_at.tzinfo is UTC


def test_track_row_contents():
    row = dict(zip(COLUMNS[TRACKS], track_row(CAMERA, SESSION, make_track(), "gap"), strict=True))
    assert row["close_reason"] == "gap"
    assert row["first_box"] == [1.0, 2.0, 3.0, 4.0]
    assert row["conf_avg"] == pytest.approx(0.8)
    assert row["path_length_px"] == 123.0


# --- запись ---------------------------------------------------------------------------


def test_flush_inserts_each_table_with_its_columns_and_database():
    sink, client, _ = make_sink(database="cams")
    sink.add(EVENTS, ["e1"])
    sink.add(POINTS, ["p1"])
    assert sink.flush() is True
    by_table = {table: (rows, columns, db) for table, rows, columns, db in client.inserts}
    assert by_table[EVENTS] == ([["e1"]], COLUMNS[EVENTS], "cams")
    assert by_table[POINTS] == ([["p1"]], COLUMNS[POINTS], "cams")
    assert sink.pending() == 0


def test_flush_with_nothing_to_write_does_not_touch_the_database():
    sink, client, factory_calls = make_sink()
    assert sink.flush() is True
    assert factory_calls == [] and client.inserts == []


def test_rows_stay_in_the_buffer_while_the_database_is_down_and_go_out_later():
    sink, client, _ = make_sink()
    client.fail_inserts = True
    sink.add(EVENTS, ["e1"])
    sink.add(EVENTS, ["e2"])
    assert sink.flush() is False
    assert sink.pending() == 2

    client.fail_inserts = False
    sink.add(EVENTS, ["e3"])
    assert sink.flush() is True
    [(table, rows, _, _)] = client.inserts
    assert (table, rows) == (EVENTS, [["e1"], ["e2"], ["e3"]])  # порядок сохранён


def test_client_is_recreated_after_a_failure():
    sink, client, factory_calls = make_sink()
    client.fail_inserts = True
    sink.add(EVENTS, ["e1"])
    sink.flush()
    client.fail_inserts = False
    sink.flush()
    assert len(factory_calls) == 2


def test_connection_error_keeps_rows_too():
    def refuse(_):
        raise ConnectionError("нет связи")

    sink = ClickHouseSink(ClickHouseConfig(create_schema=False), client_factory=refuse)
    sink.add(EVENTS, ["e1"])
    assert sink.flush() is False
    assert sink.pending() == 1


def test_only_unsent_tables_are_kept_when_a_later_table_fails():
    class FailOnPoints(FakeClient):
        def insert(self, table, rows, column_names=None, database=None):
            if table == POINTS:
                raise ConnectionError("сбой")
            super().insert(table, rows, column_names, database)

    sink, client, _ = make_sink(FailOnPoints())
    sink.add(EVENTS, ["e1"])
    sink.add(POINTS, ["p1"])
    assert sink.flush() is False
    assert [table for table, *_ in client.inserts] == [EVENTS]
    assert sink.pending() == 1  # осталась только таблица, до которой не дошло
    sink.flush()  # повтор: событие второй раз не пишется
    assert [table for table, *_ in client.inserts] == [EVENTS]


def test_schema_is_created_once_before_the_first_insert():
    sink, client, _ = make_sink(create_schema=True)
    sink.add(EVENTS, ["e1"])
    sink.flush()
    sink.add(EVENTS, ["e2"])
    sink.flush()
    assert len(client.commands) == 4  # схема применена один раз
    assert len(client.inserts) == 2


def test_failed_schema_creation_is_retried_and_rows_are_kept():
    class FlakySchema(FakeClient):
        def __init__(self):
            super().__init__()
            self.broken = True

        def command(self, sql):
            if self.broken:
                raise ConnectionError("нет связи")
            super().command(sql)

    client = FlakySchema()
    sink, _, _ = make_sink(client, create_schema=True)
    sink.add(EVENTS, ["e1"])
    assert sink.flush() is False
    assert sink.pending() == 1
    client.broken = False
    assert sink.flush() is True
    assert client.inserts


def test_buffer_limit_drops_points_first_then_summaries_then_events():
    sink, _, _ = make_sink(batch_size=1, max_buffer_rows=4)
    sink.add(EVENTS, ["e1"])
    sink.add(TRACKS, ["t1"])
    for n in range(4):
        sink.add(POINTS, [f"p{n}"])
    assert sink.pending() == 4
    assert [list(b) for b in (sink._buffers[EVENTS], sink._buffers[TRACKS])] == [[["e1"]], [["t1"]]]
    assert list(sink._buffers[POINTS]) == [["p2"], ["p3"]]  # выброшены самые старые точки


def test_full_batch_wakes_the_writer():
    sink, _, _ = make_sink(batch_size=2)
    sink.add(EVENTS, ["e1"])
    assert not sink._wake.is_set()
    sink.add(EVENTS, ["e2"])
    assert sink._wake.is_set()


def test_stop_writes_what_is_left():
    sink, client, _ = make_sink(flush_interval_sec=60, batch_size=100)
    sink.start()
    sink.add(EVENTS, ["e1"])
    sink.stop()
    assert client.inserts and client.inserts[0][1] == [["e1"]]


def test_writer_thread_flushes_by_itself():
    sink, client, _ = make_sink(flush_interval_sec=0.05, batch_size=100)
    sink.start()
    try:
        sink.add(EVENTS, ["e1"])
        deadline = time.monotonic() + 3
        while not client.inserts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.inserts
    finally:
        sink.stop()


def test_recent_events_passes_the_limit_as_a_parameter():
    client = FakeClient()
    rows = recent_events(client, "cams", 5)
    sql, parameters = client.queries[0]
    assert "cams.events" in sql and "{n:UInt32}" in sql
    assert parameters == {"n": 5}
    assert rows == [("строка",)]


def test_flushes_never_run_concurrently():
    """stop() может вызвать flush(), пока фоновый поток ещё пишет: записи идут по очереди."""
    import threading

    class SlowClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def insert(self, table, rows, column_names=None, database=None):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            super().insert(table, rows, column_names, database)
            with self.guard:
                self.active -= 1

    sink, client, _ = make_sink(SlowClient())
    sink.add(EVENTS, ["e1"])
    first = threading.Thread(target=sink.flush)
    first.start()
    time.sleep(0.01)
    sink.add(EVENTS, ["e2"])
    sink.flush()
    first.join()
    assert client.max_active == 1
    assert sorted(row for _, rows, _, _ in client.inserts for row in rows) == [["e1"], ["e2"]]


def test_schema_deduplicates_retried_inserts():
    """Повтор после оборванной вставки не должен навсегда удваивать строки."""
    for statement in schema_statements("rec")[1:]:
        assert "ENGINE = ReplacingMergeTree" in statement


def test_retry_delay_grows_and_is_capped():
    assert retry_delay(2.0, 0) == 0.0
    assert retry_delay(2.0, 1) == 4.0
    assert retry_delay(2.0, 3) == 16.0
    assert retry_delay(2.0, 4) == 30.0


def test_retry_delay_survives_thousands_of_failures():
    """База недоступна много часов: раньше 2**failures переполнял float и ронял поток записи."""
    assert retry_delay(2.0, 5000) == 30.0
