"""Socket.IO connection auth + training-room ownership (IDOR fix).

Builds a minimal Flask app and registers only the Socket.IO handlers under
test (mirroring tests/core/test_global_error_handlers.py), avoiding the heavy
create_app() boot (APScheduler, Supabase clients, etc.).

Covers:
  (a) connect with NO / invalid token  -> connection rejected (not connected)
  (b) connect with a VALID token       -> connected
  (c) join_training_session for a session NOT owned -> emits error, no join
"""
import os

os.environ.setdefault('FLASK_ENV', 'testing')

import pytest
from flask import Flask
from flask_socketio import SocketIO

import core.socketio_handlers as handlers


@pytest.fixture
def app_socketio():
    app = Flask(__name__)
    app.config['TESTING'] = True
    socketio = SocketIO(app, async_mode='threading', always_connect=True)
    handlers.register_socketio_handlers(socketio)
    return app, socketio


def _valid_token_payload(monkeypatch, sub='u1'):
    monkeypatch.setattr(handlers, '_verify_jwt_local', lambda token: {'sub': sub})


def test_connect_without_token_rejected(app_socketio):
    app, socketio = app_socketio
    client = socketio.test_client(app)  # no auth payload -> no token
    assert client.is_connected() is False


def test_connect_with_invalid_token_rejected(app_socketio, monkeypatch):
    app, socketio = app_socketio

    def _raise(token):
        raise ValueError("bad signature")

    monkeypatch.setattr(handlers, '_verify_jwt_local', _raise)
    client = socketio.test_client(app, auth={'token': 'garbage'})
    assert client.is_connected() is False


def test_connect_with_valid_token_connected(app_socketio, monkeypatch):
    app, socketio = app_socketio
    _valid_token_payload(monkeypatch, sub='u1')

    client = socketio.test_client(app, auth={'token': 'valid.jwt.token'})
    assert client.is_connected() is True
    client.disconnect()


def test_join_training_session_not_owned_emits_error_and_no_join(app_socketio, monkeypatch):
    app, socketio = app_socketio
    _valid_token_payload(monkeypatch, sub='u1')

    def _deny(session_id, user_id=None, create_if_missing=True):
        raise PermissionError(f"Session {session_id} does not belong to user")

    monkeypatch.setattr(handlers, 'create_or_get_session_uuid', _deny)

    client = socketio.test_client(app, auth={'token': 'valid.jwt.token'})
    assert client.is_connected() is True

    client.get_received()  # drain anything queued on connect
    client.emit('join_training_session', {'session_id': 'sess-not-owned'})
    received = client.get_received()

    events = {r['name'] for r in received}
    assert 'training_session_error' in events
    assert 'training_session_joined' not in events
    client.disconnect()


def test_join_training_session_owned_succeeds(app_socketio, monkeypatch):
    app, socketio = app_socketio
    _valid_token_payload(monkeypatch, sub='u1')

    monkeypatch.setattr(
        handlers, 'create_or_get_session_uuid',
        lambda session_id, user_id=None, create_if_missing=True: 'uuid-1234',
    )

    client = socketio.test_client(app, auth={'token': 'valid.jwt.token'})
    assert client.is_connected() is True

    client.get_received()
    client.emit('join_training_session', {'session_id': 'sess-owned'})
    received = client.get_received()

    events = {r['name'] for r in received}
    assert 'training_session_joined' in events
    assert 'training_session_error' not in events
    client.disconnect()
