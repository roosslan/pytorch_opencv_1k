import pytest

from rectrack.config import PASSWORD_ENV, ConfigError, load_config, parse_config, redact_url

BASE = {"camera": {"rtsp_url": "rtsp://cam/1"}}


def raw(**sections):
    """Минимально допустимая конфигурация плюс переопределённые разделы."""
    merged = {name: dict(values) for name, values in BASE.items()}
    for name, values in sections.items():
        merged.setdefault(name, {}).update(values)
    return merged


def test_defaults_are_applied():
    cfg = parse_config(BASE)
    assert cfg.camera.name == "cam1"
    assert cfg.model.classes == ["person"]
    assert cfg.model.device == "cpu"
    assert cfg.tracking.confirm_hits == 3
    assert cfg.clickhouse.port == 8123
    assert cfg.clickhouse.database == "rec"


def test_values_are_overridden():
    cfg = parse_config(
        raw(
            model={"device": "cuda:0", "classes": ["person", "car"], "half": True},
            clickhouse={"host": "ch.local", "database": "cams"},
        )
    )
    assert cfg.model.classes == ["person", "car"]
    assert cfg.model.half is True
    assert (cfg.clickhouse.host, cfg.clickhouse.database) == ("ch.local", "cams")


def test_empty_class_list_means_all_classes():
    assert parse_config(raw(model={"classes": []})).model.classes == []


def test_int_is_accepted_for_float_field():
    assert parse_config(raw(tracking={"lost_timeout_sec": 5})).tracking.lost_timeout_sec == 5


def test_rtsp_url_is_required():
    with pytest.raises(ConfigError, match="rtsp_url"):
        parse_config({})


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("camera", "transport", "http", "transport"),
        ("camera", "timeout_sec", 0, "timeout_sec"),
        ("model", "imgsz", 8, "imgsz"),
        ("model", "conf", 0, "conf"),
        ("model", "conf", 1, "conf"),
        ("model", "iou", 0, "iou"),
        ("model", "weights", "", "weights"),
        ("tracking", "confirm_hits", 0, "confirm_hits"),
        ("tracking", "lost_timeout_sec", 0, "lost_timeout_sec"),
        ("tracking", "points_interval_sec", -1, "points_interval_sec"),
        ("tracking", "trail_length", 0, "trail_length"),
        ("snapshots", "directory", "", "directory"),
        ("clickhouse", "port", 70000, "port"),
        ("clickhouse", "database", "rec; DROP TABLE x", "database"),
        ("clickhouse", "database", "1rec", "database"),
        ("clickhouse", "batch_size", 0, "batch_size"),
        ("clickhouse", "flush_interval_sec", 0, "flush_interval_sec"),
        ("clickhouse", "max_buffer_rows", 10, "max_buffer_rows"),  # меньше batch_size
    ],
)
def test_invalid_values_are_rejected(section, key, value, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw(**{section: {key: value}}))


def test_proxy_is_empty_by_default_and_can_be_set():
    assert parse_config(BASE).camera.proxy == ""
    cfg = parse_config(raw(camera={"proxy": "http://192.168.1.20:8888"}))
    assert cfg.camera.proxy == "http://192.168.1.20:8888"


@pytest.mark.parametrize(
    "camera, message",
    [
        ({"proxy": "192.168.1.20:8888"}, "proxy"),
        ({"proxy": "http://192.168.1.20"}, "proxy"),
        ({"proxy": "http://192.168.1.20:8888", "transport": "udp"}, "tcp"),
        ({"proxy": "http://192.168.1.20:8888", "rtsp_url": "http://cam/1"}, "rtsp_url"),
    ],
)
def test_invalid_proxy_settings_are_rejected(camera, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(raw(camera=camera))


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="confidence"):
        parse_config(raw(model={"confidence": 0.5}))


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="tracing"):
        parse_config(raw(tracing={}))


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="port"):
        parse_config(raw(clickhouse={"port": "8123"}))


def test_bool_is_not_accepted_as_int():
    with pytest.raises(ConfigError, match="port"):
        parse_config(raw(clickhouse={"port": True}))


def test_class_list_must_contain_strings():
    with pytest.raises(ConfigError, match="списком строк"):
        parse_config(raw(model={"classes": ["person", 0]}))


def test_load_config_reads_toml_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[camera]\nrtsp_url = "rtsp://cam/2"\nname = "вход"\n', encoding="utf-8")
    cfg = load_config(path)
    assert (cfg.camera.rtsp_url, cfg.camera.name) == ("rtsp://cam/2", "вход")


def test_load_config_missing_file_points_to_example(tmp_path):
    with pytest.raises(ConfigError, match=r"config\.example\.toml"):
        load_config(tmp_path / "nope.toml")


def test_load_config_invalid_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[camera\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Некорректный TOML"):
        load_config(path)


def test_password_from_environment_beats_the_file(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[camera]\nrtsp_url = "x"\n[clickhouse]\npassword = "from-file"\n', encoding="utf-8"
    )
    monkeypatch.setenv(PASSWORD_ENV, "from-env")
    assert load_config(path).clickhouse.password == "from-env"


def test_password_from_file_is_used_without_environment(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[camera]\nrtsp_url = "x"\n[clickhouse]\npassword = "from-file"\n', encoding="utf-8"
    )
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    assert load_config(path).clickhouse.password == "from-file"


def test_redact_url_hides_credentials():
    assert redact_url("rtsp://admin:secret@10.0.0.5:554/1") == "rtsp://***@10.0.0.5:554/1"


def test_redact_url_keeps_plain_url():
    assert redact_url("rtsp://10.0.0.5:554/1") == "rtsp://10.0.0.5:554/1"


def test_password_from_dotenv_next_to_the_config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[camera]\nrtsp_url = "x"\n[clickhouse]\npassword = "from-file"\n', encoding="utf-8"
    )
    (tmp_path / ".env").write_text(
        f"# комментарий\nCLICKHOUSE_PASSWORD=other\n{PASSWORD_ENV}=\"from-dotenv\"\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    assert load_config(path).clickhouse.password == "from-dotenv"
    monkeypatch.setenv(PASSWORD_ENV, "from-env")  # окружение главнее .env
    assert load_config(path).clickhouse.password == "from-env"


def test_empty_password_in_dotenv_keeps_the_file_value(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[camera]\nrtsp_url = "x"\n[clickhouse]\npassword = "from-file"\n', encoding="utf-8"
    )
    (tmp_path / ".env").write_text(f"{PASSWORD_ENV}=\n", encoding="utf-8")
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    assert load_config(path).clickhouse.password == "from-file"
