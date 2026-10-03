-- Схема ClickHouse. {database} подставляется из config.toml ([clickhouse] database).
-- Все времена хранятся в UTC.
--
-- ReplacingMergeTree: если вставка оборвалась по таймауту, но сервер её всё же принял, при
-- повторе строки придут второй раз. Одинаковые по ключу сортировки строки ClickHouse схлопывает
-- при слиянии частей; до слияния дубли видны, точный результат даёт SELECT ... FINAL.

CREATE DATABASE IF NOT EXISTS {database};

-- Журнал событий: захват цели и её потеря.
CREATE TABLE IF NOT EXISTS {database}.events
(
    ts            DateTime64(3, 'UTC'),
    camera        LowCardinality(String),
    event         LowCardinality(String),   -- 'captured' или 'lost'
    session_id    UUID,                     -- один запуск программы
    track_uid     UUID,                     -- уникальный номер цели
    track_id      UInt32,                   -- номер цели от трекера внутри запуска
    class_name    LowCardinality(String),
    conf          Float32,
    duration_sec  Float32,                  -- для 'lost': сколько цель была в кадре
    snapshot      String                    -- путь к снимку, сделанному при захвате
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (camera, ts, track_uid, event);

-- Итог по каждой цели: одна строка на цель, пишется, когда сопровождение закончено.
CREATE TABLE IF NOT EXISTS {database}.tracks
(
    camera          LowCardinality(String),
    session_id      UUID,
    track_uid       UUID,
    track_id        UInt32,
    class_name      LowCardinality(String),
    captured_at     DateTime64(3, 'UTC'),
    last_seen       DateTime64(3, 'UTC'),
    duration_sec    Float32,
    hits            UInt32,                 -- на скольких кадрах цель была видна
    conf_max        Float32,
    conf_avg        Float32,
    path_length_px  Float32,                -- пройденный путь в пикселях кадра
    first_box       Array(Float32),         -- x1, y1, x2, y2 при первом появлении
    last_box        Array(Float32),         -- x1, y1, x2, y2 при последнем появлении
    snapshot        String,
    close_reason    LowCardinality(String)  -- 'lost', 'gap' (разрыв потока) или 'shutdown'
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(captured_at)
ORDER BY (camera, captured_at, track_uid);

-- Траектории: точки положения цели, реже чем кадры (см. points_interval_sec).
-- Хранятся 30 дней, потом ClickHouse удаляет их сам.
CREATE TABLE IF NOT EXISTS {database}.track_points
(
    ts         DateTime64(3, 'UTC'),
    camera     LowCardinality(String),
    track_uid  UUID,
    track_id   UInt32,
    x1         Float32,
    y1         Float32,
    x2         Float32,
    y2         Float32,
    conf       Float32
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY (camera, track_uid, ts)
TTL toDateTime(ts) + INTERVAL 30 DAY;
