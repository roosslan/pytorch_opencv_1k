"""Доступ к RTSP-камере через HTTP-прокси с методом CONNECT (например, tinyproxy).

OpenCV и FFmpeg не умеют подключаться к RTSP через HTTP-прокси. Поэтому программа поднимает на
127.0.0.1 локальный туннель: каждое входящее соединение пробрасывается через прокси
(`CONNECT камера:порт`) до камеры, а FFmpeg подключается к туннелю как к самой камере.

Через туннель работает только `transport = "tcp"`: видео идёт внутри того же TCP-соединения,
что и команды RTSP, а UDP прокси не передаёт.
"""

import base64
import logging
import socket
import threading
from collections.abc import Callable
from urllib.parse import unquote, urlsplit, urlunsplit

log = logging.getLogger(__name__)

RTSP_DEFAULT_PORT = 554
LOCAL_HOST = "127.0.0.1"
_MAX_HEAD_BYTES = 16 * 1024
_BUFFER_BYTES = 64 * 1024


class ProxyError(ConnectionError):
    """Прокси недоступен, закрыл соединение или отказал в CONNECT."""


def parse_proxy(proxy: str) -> tuple[str, int, str | None]:
    """Разбирает адрес прокси вида http://адрес:порт (или http://логин:пароль@адрес:порт).

    Args:
        proxy: адрес прокси из конфигурации.

    Returns:
        кортеж (адрес, порт, значение заголовка Proxy-Authorization или None, если логина
        в адресе нет).

    Raises:
        ValueError: схема не http, нет адреса или порта, либо порт некорректен.
    """
    parts = urlsplit(proxy)
    if parts.scheme != "http":
        raise ValueError("адрес прокси должен начинаться с http://")
    if not parts.hostname:
        raise ValueError("в адресе прокси нет хоста")
    if parts.port is None:  # некорректный порт urlsplit сам превращает в ValueError
        raise ValueError("в адресе прокси нужно указать порт, например http://192.168.1.20:8888")
    auth = None
    if parts.username is not None:
        credentials = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
        auth = "Basic " + base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    return parts.hostname, parts.port, auth


def rtsp_target(url: str) -> tuple[str, int]:
    """Извлекает из адреса камеры хост и порт, к которым прокси должен открыть соединение.

    Args:
        url: адрес потока, например rtsp://user:pass@192.168.1.10:554/1.

    Returns:
        кортеж (хост, порт); если порт в адресе не указан, берётся стандартный 554.

    Raises:
        ValueError: схема не rtsp, нет хоста или порт некорректен.
    """
    parts = urlsplit(url)
    if parts.scheme != "rtsp":
        raise ValueError("через прокси поддерживаются только адреса rtsp://")
    if not parts.hostname:
        raise ValueError("в адресе камеры нет хоста")
    return parts.hostname, parts.port or RTSP_DEFAULT_PORT


def local_url(url: str, port: int) -> str:
    """Заменяет в адресе камеры хост и порт на локальный туннель; логин, пароль, путь и
    параметры запроса сохраняются.

    Args:
        url: исходный адрес камеры.
        port: порт локального туннеля на 127.0.0.1.

    Returns:
        адрес вида rtsp://[логин:пароль@]127.0.0.1:порт/путь.
    """
    parts = urlsplit(url)
    userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    return urlunsplit(parts._replace(netloc=f"{userinfo}{LOCAL_HOST}:{port}"))


def open_tunnel(
    proxy_host: str,
    proxy_port: int,
    auth: str | None,
    target_host: str,
    target_port: int,
    timeout: float,
) -> tuple[socket.socket, bytes]:
    """Подключается к прокси и просит его открыть соединение с камерой (метод CONNECT).

    Args:
        proxy_host: адрес прокси.
        proxy_port: порт прокси.
        auth: значение заголовка Proxy-Authorization или None, если авторизации нет.
        target_host: адрес камеры.
        target_port: порт RTSP камеры.
        timeout: таймаут подключения к прокси и ожидания его ответа в секундах.

    Returns:
        кортеж (сокет, байты): сокет уже соединён с камерой через прокси и работает без
        таймаута; байты - то, что пришло от камеры сразу после ответа прокси (обычно пусто).

    Raises:
        ProxyError: прокси закрыл соединение, прислал слишком длинный или непонятный ответ
            либо отказал (код ответа не 200).
        OSError: прокси недоступен или не ответил за timeout.
    """
    sock = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    try:
        host = f"[{target_host}]" if ":" in target_host else target_host  # IPv6 в скобках
        target = f"{host}:{target_port}"
        request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n"
        if auth:
            request += f"Proxy-Authorization: {auth}\r\n"
        sock.sendall((request + "\r\n").encode("ascii"))

        data = b""
        while b"\r\n\r\n" not in data:
            if len(data) > _MAX_HEAD_BYTES:
                raise ProxyError("слишком длинный ответ прокси")
            chunk = sock.recv(4096)
            if not chunk:
                raise ProxyError("прокси закрыл соединение, не ответив на CONNECT")
            data += chunk
        head, rest = data.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].decode("latin-1")
        fields = status.split(None, 2)
        if len(fields) < 2 or not fields[0].startswith("HTTP/"):
            raise ProxyError(f"непонятный ответ прокси: {status!r}")
        if fields[1] != "200":
            hint = ""
            if fields[1] == "403":
                hint = f" (для tinyproxy: в tinyproxy.conf нужна строка ConnectPort {target_port})"
            raise ProxyError(f"прокси отказал в соединении с {target}: {status}{hint}")
        sock.settimeout(None)  # дальше таймауты чтения соблюдает FFmpeg на своей стороне
        return sock, rest
    except BaseException:
        sock.close()
        raise


class ProxyTunnel:
    """Локальный TCP-туннель 127.0.0.1:порт -> HTTP-прокси -> камера."""

    def __init__(self, proxy: str, target_host: str, target_port: int, timeout: float = 5.0):
        """Запоминает настройки; слушать порт туннель начинает только в start().

        Args:
            proxy: адрес прокси вида http://адрес:порт.
            target_host: адрес камеры, к которой прокси открывает соединение.
            target_port: порт RTSP камеры.
            timeout: таймаут подключения к прокси и ожидания его ответа в секундах.

        Raises:
            ValueError: адрес прокси некорректен (см. parse_proxy).
        """
        self._proxy_host, self._proxy_port, self._auth = parse_proxy(proxy)
        self._target_host = target_host
        self._target_port = target_port
        self._timeout = timeout
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        self.port = 0  # локальный порт; известен после start()

    def start(self) -> None:
        """Открывает локальный порт (выбирается системой, только на 127.0.0.1, поэтому
        снаружи к туннелю не подключиться) и запускает фоновый поток приёма соединений."""
        server = socket.create_server((LOCAL_HOST, 0))
        server.settimeout(0.5)  # чтобы поток приёма вовремя замечал stop()
        self._server = server
        self.port = server.getsockname()[1]
        threading.Thread(target=self._accept_loop, name="rtsp-proxy", daemon=True).start()

    def stop(self) -> None:
        """Перестаёт принимать соединения и закрывает все открытые через туннель."""
        self._stop.set()
        if self._server is not None:
            self._server.close()
        with self._lock:
            sockets, self._sockets = list(self._sockets), set()
        for sock in sockets:
            _close(sock)

    def _accept_loop(self) -> None:
        """Тело фонового потока: принимает соединения FFmpeg и для каждого запускает
        отдельный поток, который пробрасывает его через прокси. Работает до stop()."""
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break  # сокет закрыт в stop()
            client.settimeout(None)
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        """Открывает туннель через прокси для одного соединения FFmpeg и гоняет данные
        в обе стороны, пока одна из сторон не закроет соединение. Если прокси недоступен
        или отказал, пишет предупреждение и закрывает соединение FFmpeg: тот увидит ошибку
        и переподключится сам.

        Args:
            client: принятое соединение от FFmpeg.
        """
        try:
            upstream, rest = open_tunnel(
                self._proxy_host,
                self._proxy_port,
                self._auth,
                self._target_host,
                self._target_port,
                self._timeout,
            )
        except OSError as exc:
            log.warning("Прокси %s:%d: %s", self._proxy_host, self._proxy_port, exc)
            _close(client)
            return
        with self._lock:
            if self._stop.is_set():
                _close(client)
                _close(upstream)
                return
            self._sockets.update((client, upstream))

        done = threading.Event()

        def finish() -> None:
            if done.is_set():
                return
            done.set()
            with self._lock:
                self._sockets.difference_update((client, upstream))
            _close(client)
            _close(upstream)

        try:
            if rest:
                client.sendall(rest)
        except OSError:
            finish()
            return
        threading.Thread(target=_pipe, args=(upstream, client, finish), daemon=True).start()
        _pipe(client, upstream, finish)


def _pipe(src: socket.socket, dst: socket.socket, finish: Callable[[], None]) -> None:
    """Копирует данные из src в dst, пока src не закроется или не случится ошибка; затем
    вызывает finish(), который закрывает обе стороны.

    Args:
        src: сокет, из которого читать.
        dst: сокет, в который писать.
        finish: функция без аргументов, закрывающая соединение целиком.
    """
    try:
        while data := src.recv(_BUFFER_BYTES):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        finish()


def _close(sock: socket.socket) -> None:
    """Закрывает сокет. shutdown() перед close() будит поток, который ждёт в recv() на этом
    сокете (на Linux один close() этого не делает). Ошибки игнорируются.

    Args:
        sock: сокет, который нужно закрыть.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
