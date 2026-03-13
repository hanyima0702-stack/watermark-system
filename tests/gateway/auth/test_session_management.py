"""Tests for user session management."""

import pytest
import hashlib
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta

from gateway.api.auth.auth_service import AuthService
from gateway.api.auth.jwt_manager import JWTManager
from gateway.api.auth.password_utils import hash_password
from storage.models.user import User
from storage.models.user_session import UserSession


@pytest.fixture
def jwt_manager():
    return JWTManager(secret_key="test-secret-key", algorithm="HS256", expire_minutes=30)


@pytest.fixture
def mock_user_dao():
    return AsyncMock()


@pytest.fixture
def mock_session_dao():
    return AsyncMock()


@pytest.fixture
def mock_db_manager(mock_user_dao, mock_session_dao):
    db_manager = Mock()
    def get_dao(name):
        if name == 'user':
            return mock_user_dao
        if name == 'user_session':
            return mock_session_dao
        raise ValueError(f"Unknown DAO: {name}")
    db_manager.get_dao = Mock(side_effect=get_dao)
    return db_manager


@pytest.fixture
async def auth_service(mock_db_manager, jwt_manager):
    service = AuthService(mock_db_manager, jwt_manager)
    await service.initialize()
    return service


@pytest.fixture
def sample_user():
    return User(
        user_id="user-123",
        username="testuser",
        email="test@example.com",
        department="IT",
        password_hash=hash_password("Password123"),
        salt="",
        roles=["user"],
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


class TestSessionRecording:
    """Tests for session recording on login (Requirement 7.1)."""

    @pytest.mark.asyncio
    async def test_login_creates_session(self, auth_service, mock_user_dao, mock_session_dao, sample_user):
        """Login should create a session record."""
        mock_user_dao.get_by_username.return_value = sample_user
        mock_user_dao.update.return_value = sample_user
        mock_session_dao.create.return_value = UserSession(
            id="session-1", user_id=sample_user.user_id,
            token_hash="hash", expires_at=datetime.utcnow() + timedelta(hours=1),
            is_active=True
        )

        user, token = await auth_service.login(
            username="testuser",
            password="Password123",
            ip_address="192.168.1.1",
            user_agent="TestBrowser/1.0"
        )

        assert user is not None
        assert token is not None
        mock_session_dao.create.assert_called_once()

        created_session = mock_session_dao.create.call_args[0][0]
        assert created_session.user_id == sample_user.user_id
        assert created_session.ip_address == "192.168.1.1"
        assert created_session.user_agent == "TestBrowser/1.0"
        assert created_session.is_active is True

    @pytest.mark.asyncio
    async def test_login_session_has_correct_token_hash(self, auth_service, mock_user_dao, mock_session_dao, sample_user):
        """Session should store a SHA-256 hash of the JWT token."""
        mock_user_dao.get_by_username.return_value = sample_user
        mock_user_dao.update.return_value = sample_user
        mock_session_dao.create.return_value = UserSession(
            id="session-1", user_id=sample_user.user_id,
            token_hash="hash", expires_at=datetime.utcnow() + timedelta(hours=1),
            is_active=True
        )

        _, token = await auth_service.login(username="testuser", password="Password123")

        created_session = mock_session_dao.create.call_args[0][0]
        expected_hash = hashlib.sha256(token.encode()).hexdigest()
        assert created_session.token_hash == expected_hash

    @pytest.mark.asyncio
    async def test_login_session_without_client_info(self, auth_service, mock_user_dao, mock_session_dao, sample_user):
        """Login without IP/UA should still create a session with None values."""
        mock_user_dao.get_by_username.return_value = sample_user
        mock_user_dao.update.return_value = sample_user
        mock_session_dao.create.return_value = UserSession(
            id="session-1", user_id=sample_user.user_id,
            token_hash="hash", expires_at=datetime.utcnow() + timedelta(hours=1),
            is_active=True
        )

        await auth_service.login(username="testuser", password="Password123")

        created_session = mock_session_dao.create.call_args[0][0]
        assert created_session.ip_address is None
        assert created_session.user_agent is None


class TestSessionDeactivation:
    """Tests for session deactivation on logout (Requirement 7.2)."""

    @pytest.mark.asyncio
    async def test_logout_deactivates_session(self, auth_service, mock_user_dao, mock_session_dao, sample_user, jwt_manager):
        """Logout should deactivate the session by token hash."""
        token = jwt_manager.create_access_token(user_id=sample_user.user_id, username=sample_user.username)
        mock_user_dao.get_by_id.return_value = sample_user
        mock_session_dao.deactivate_by_token_hash.return_value = True

        result = await auth_service.logout(token)

        assert result is True
        expected_hash = hashlib.sha256(token.encode()).hexdigest()
        mock_session_dao.deactivate_by_token_hash.assert_called_once_with(expected_hash)

    @pytest.mark.asyncio
    async def test_logout_invalid_token(self, auth_service):
        """Logout with invalid token should return False."""
        result = await auth_service.logout("invalid.token.here")
        assert result is False


class TestSessionQuery:
    """Tests for session query (Requirement 7.4)."""

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, auth_service, mock_session_dao):
        """Should return active sessions for a user."""
        sessions = [
            UserSession(
                id="s1", user_id="user-123", token_hash="h1",
                expires_at=datetime.utcnow() + timedelta(hours=1),
                ip_address="10.0.0.1", user_agent="Chrome", is_active=True
            ),
            UserSession(
                id="s2", user_id="user-123", token_hash="h2",
                expires_at=datetime.utcnow() + timedelta(hours=2),
                ip_address="10.0.0.2", user_agent="Firefox", is_active=True
            ),
        ]
        mock_session_dao.get_active_sessions_by_user.return_value = sessions

        result = await auth_service.get_active_sessions("user-123")

        assert len(result) == 2
        mock_session_dao.get_active_sessions_by_user.assert_called_once_with("user-123")


class TestSessionsEndpoint:
    """Tests for GET /api/v1/auth/sessions endpoint."""

    @pytest.mark.asyncio
    async def test_get_sessions_returns_active_sessions(self, auth_service, mock_session_dao):
        """Should return active sessions for the authenticated user."""
        now = datetime.utcnow()
        sessions = [
            UserSession(
                id="s1", user_id="user-123", token_hash="h1",
                expires_at=now + timedelta(hours=1),
                created_at=now, last_activity_at=now,
                ip_address="10.0.0.1", user_agent="Chrome/120", is_active=True
            ),
        ]
        mock_session_dao.get_active_sessions_by_user.return_value = sessions

        result = await auth_service.get_active_sessions("user-123")

        assert len(result) == 1
        assert result[0].id == "s1"
        assert result[0].ip_address == "10.0.0.1"
        assert result[0].user_agent == "Chrome/120"

    @pytest.mark.asyncio
    async def test_get_sessions_empty(self, auth_service, mock_session_dao):
        """Should return empty list when no active sessions."""
        mock_session_dao.get_active_sessions_by_user.return_value = []

        result = await auth_service.get_active_sessions("user-123")

        assert result == []
