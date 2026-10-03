import base64
import socket
import threading

import pytest

from rectrack.proxy import (
    ProxyError,
    ProxyTunnel,
    local_url,
    open_tunnel,
    parse_proxy,
    rtsp_target,
)


class FakeProxy:
    """HTTP-прокси для тестов: отвечает на CONNECT, как tinyproxy, а затем сам изображает
    камеру - возвращает всё, что получил, с префиксом "echo:"."""

    def __init__(self, status: str = "200 Connection established", extra: bytes = b""):
        self.status = status
        self.extra = extra  # байты «от камеры», пришедшие вместе с ответом прокси
        self.requests: list[str] = []
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(1024)
                if not chunk:
                    return
                data += chunk
            self.requests.append(data.decode("ascii"))
            reply = f"HTTP/1.0 {self.status}\r\nProxy-agent: tinyproxy/1.11\r\n\r\n".encode()
            conn.sendall(reply + self.extra)
            if not self.status.startswith("200"):
                return
            while chunk := conn.recv(1024):
                conn.sendall(b"echo:" + chunk)

    def close(self):
        self.server.close()


@pytest.fixture
def proxy():
    fake = FakeProxy()
    yield fake
    fake.close()


def recv_exactly(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def test_parse_proxy_without_auth():
    assert parse_proxy("http://192.168.1.20:8888") == ("192.168.1.20", 8888, None)


def test_parse_proxy_with_auth():
    host, port, auth = parse_proxy("http://user:p%40ss@proxy.lan:3128")
    assert (host, port) == ("proxy.lan", 3128)
    assert auth == "Basic " + base64.b64encode(b"user:p@ss").decode()


@pytest.mark.parametrize(
    "value", ["192.168.1.20:8888", "https://proxy.lan:8888", "http://proxy.lan", "http://:8888"]
)
def test_parse_proxy_rejects_bad_addresses(value):
    with pytest.raises(ValueError):
        parse_proxy(value)


def test_rtsp_target_default_and_explicit_port():
    assert rtsp_target("rtsp://192.168.1.10/1") == ("192.168.1.10", 554)
    assert rtsp_target("rtsp://u:p@192.168.1.10:8554/1") == ("192.168.1.10", 8554)


def test_rtsp_target_rejects_other_schemes():
    with pytest.raises(ValueError):
        rtsp_target("http://192.168.1.10/1")


def test_local_url_keeps_credentials_path_and_query():
    url = "rtsp://admin:secret@192.168.1.10:554/Streaming/Channels/101?transport=tcp"
    assert local_url(url, 40000) == (
        "rtsp://admin:secret@127.0.0.1:40000/Streaming/Channels/101?transport=tcp"
    )
    assert local_url("rtsp://192.168.1.10/1", 40000) == "rtsp://127.0.0.1:40000/1"


def test_open_tunnel_sends_connect_and_relays(proxy):
    sock, rest = open_tunnel("127.0.0.1", proxy.port, None, "192.168.1.10", 554, 2.0)
    with sock:
        assert rest == b""
        assert proxy.requests[0].startswith("CONNECT 192.168.1.10:554 HTTP/1.1\r\n")
        assert "Host: 192.168.1.10:554\r\n" in proxy.requests[0]
        assert "Proxy-Authorization" not in proxy.requests[0]
        sock.sendall(b"OPTIONS")
        assert recv_exactly(sock, 12) == b"echo:OPTIONS"


def test_open_tunnel_sends_auth_header(proxy):
    _, _, auth = parse_proxy("http://user:pass@127.0.0.1:1")
    sock, _ = open_tunnel("127.0.0.1", proxy.port, auth, "cam.lan", 554, 2.0)
    sock.close()
    assert f"Proxy-Authorization: {auth}\r\n" in proxy.requests[0]


def test_open_tunnel_returns_bytes_after_the_reply():
    fake = FakeProxy(extra=b"RTSP/1.0")
    try:
        sock, rest = open_tunnel("127.0.0.1", fake.port, None, "cam.lan", 554, 2.0)
        sock.close()
        assert rest == b"RTSP/1.0"
    finally:
        fake.close()


def test_open_tunnel_refused_with_tinyproxy_hint():
    fake = FakeProxy(status="403 Access violation")
    try:
        with pytest.raises(ProxyError, match="ConnectPort 554"):
            open_tunnel("127.0.0.1", fake.port, None, "cam.lan", 554, 2.0)
    finally:
        fake.close()


def test_open_tunnel_unreachable_proxy():
    free = socket.create_server(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    with pytest.raises(OSError):
        open_tunnel("127.0.0.1", port, None, "cam.lan", 554, 1.0)


def test_tunnel_relays_both_ways(proxy):
    tunnel = ProxyTunnel(f"http://127.0.0.1:{proxy.port}", "192.168.1.10", 554, timeout=2.0)
    tunnel.start()
    try:
        with socket.create_connection(("127.0.0.1", tunnel.port), timeout=2.0) as client:
            client.sendall(b"DESCRIBE")
            assert recv_exactly(client, 13) == b"echo:DESCRIBE"
        # FFmpeg после обрыва открывает новое соединение: туннель должен принять и его
        with socket.create_connection(("127.0.0.1", tunnel.port), timeout=2.0) as client:
            client.sendall(b"SETUP")
            assert recv_exactly(client, 10) == b"echo:SETUP"
        assert len(proxy.requests) == 2
    finally:
        tunnel.stop()


def test_tunnel_closes_client_when_proxy_refuses(caplog):
    fake = FakeProxy(status="403 Access violation")
    tunnel = ProxyTunnel(f"http://127.0.0.1:{fake.port}", "cam.lan", 554, timeout=2.0)
    tunnel.start()
    try:
        with socket.create_connection(("127.0.0.1", tunnel.port), timeout=2.0) as client:
            assert client.recv(10) == b""  # FFmpeg увидит обрыв и переподключится
        assert "ConnectPort 554" in caplog.text
    finally:
        tunnel.stop()
        fake.close()


def test_stop_closes_open_connections(proxy):
    tunnel = ProxyTunnel(f"http://127.0.0.1:{proxy.port}", "cam.lan", 554, timeout=2.0)
    tunnel.start()
    client = socket.create_connection(("127.0.0.1", tunnel.port), timeout=2.0)
    try:
        client.sendall(b"PLAY")
        assert recv_exactly(client, 9) == b"echo:PLAY"
        tunnel.stop()
        assert client.recv(10) == b""
    finally:
        client.close()
