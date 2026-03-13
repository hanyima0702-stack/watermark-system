"""Integration tests for authentication service."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from gateway.api.auth.auth_service import (
    AuthService,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
    WeakPasswordError
)
from gateway.api.auth.jwt_manager import JWTManager
from gateway.api.auth.password_utils import hash_password
from storage.models.user import User


@pytest.fixture
def jwt_manager():
    """Create JWT manager for testing."""
    return JWTManager(
        secret_key="test-secret-key",
        algorithm="HS256",
        expire_minutes=30
    )


@pytest.fixture
def mock_user_dao():
    """Create mock UserDAO."""
    dao = AsyncMock()
    return dao


@pytest.fixture
def mock_db_manager(mock_user_dao):
    """Create mock database manager."""
    db_manager = Mock()
    db_manager.get_dao = Mock(return_value=mock_user_dao)
    return db_manager


@pytest.fixture
async def auth_service(mock_db_manager, jwt_manager):
    """Create auth service instance."""
    service = AuthService(mock_db_manager, jwt_manager)
    await service.initialize()
    return service


@pytest.fixture
def sample_user():
    """Create sample user for testing."""
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


class TestUserRegistration:
    """Tests for user registration."""
    
    @pytest.mark.asyncio
    async def test_register_success(self, auth_service, mock_user_dao):
        """Test successful user registration."""
        # Setup mocks
        mock_user_dao.get_by_username.return_value = None
        mock_user_dao.get_by_email.return_value = None
        
        created_user = User(
            user_id="new-user-123",
            username="newuser",
            email="new@example.com",
            department="IT",
            password_hash="hashed",
            salt="",
            roles=["user"],
            is_active=True
        )
        mock_user_dao.create.return_value = created_user
        
        # Register user
        user, token = await auth_service.register(
            username="newuser",
            password="Password123",
            email="new@example.com",
            department="IT"
        )
        
        # Verify
        assert user is not None
        assert user.username == "newuser"
        assert token is not None
        assert len(token) > 0
        mock_user_dao.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_username_exists(self, auth_service, mock_user_dao, sample_user):
        """Test registration with existing username."""
        mock_user_dao.get_by_username.return_value = sample_user
        
        with pytest.raises(UserAlreadyExistsError, match="用户名已被使用"):
            await auth_service.register(
                username="testuser",
                password="Password123",
                email="different@example.com"
            )
    
    @pytest.mark.asyncio
    async def test_register_email_exists(self, auth_service, mock_user_dao, sample_user):
        """Test registration with existing email."""
        mock_user_dao.get_by_username.return_value = None
        mock_user_dao.get_by_email.return_value = sample_user
        
        with pytest.raises(UserAlreadyExistsError, match="邮箱已被使用"):
            await auth_service.register(
                username="differentuser",
                password="Password123",
                email="test@example.com"
            )
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self, auth_service, mock_user_dao):
        """Test registration with weak password."""
        mock_user_dao.get_by_username.return_value = None
        mock_user_dao.get_by_email.return_value = None
        
        with pytest.raises(WeakPasswordError):
            await auth_service.register(
                username="newuser",
                password="weak",
                email="new@example.com"
            )
    
    @pytest.mark.asyncio
    async def test_register_password_no_number(self, auth_service, mock_user_dao):
        """Test registration with password missing number."""
        mock_user_dao.get_by_username.return_value = None
        mock_user_dao.get_by_email.return_value = None
        
        with pytest.raises(WeakPasswordError, match="至少一个数字"):
            await auth_service.register(
                username="newuser",
                password="PasswordOnly",
                email="new@example.com"
            )


class TestUserLogin:
    """Tests for user login."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, auth_service, mock_user_dao, sample_user):
        """Test successful login."""
        mock_user_dao.get_by_username.return_value = sample_user
        mock_user_dao.update.return_value = sample_user
        
        user, token = await auth_service.login(
            username="testuser",
            password="Password123"
        )
        
        assert user is not None
        assert user.username == "testuser"
        assert token is not None
        assert len(token) > 0
        mock_user_dao.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_login_user_not_found(self, auth_service, mock_user_dao):
        """Test login with non-existent user."""
        mock_user_dao.get_by_username.return_value = None
        
        with pytest.raises(UserNotFoundError, match="用户名或密码错误"):
            await auth_service.login(
                username="nonexistent",
                password="Password123"
            )
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, auth_service, mock_user_dao, sample_user):
        """Test login with wrong password."""
        mock_user_dao.get_by_username.return_value = sample_user
        
        with pytest.raises(InvalidCredentialsError, match="用户名或密码错误"):
            await auth_service.login(
                username="testuser",
                password="WrongPassword123"
            )
    
    @pytest.mark.asyncio
    async def test_login_inactive_user(self, auth_service, mock_user_dao, sample_user):
        """Test login with inactive user."""
        sample_user.is_active = False
        mock_user_dao.get_by_username.return_value = sample_user
        
        with pytest.raises(InvalidCredentialsError, match="用户账号已被停用"):
            await auth_service.login(
                username="testuser",
                password="Password123"
            )


class TestTokenVerification:
    """Tests for token verification."""
    
    @pytest.mark.asyncio
    async def test_verify_token_success(self, auth_service, mock_user_dao, sample_user, jwt_manager):
        """Test successful token verification."""
        token = jwt_manager.create_access_token(
            user_id=sample_user.user_id,
            username=sample_user.username
        )
        
        mock_user_dao.get_by_id.return_value = sample_user
        
        user = await auth_service.verify_token(token)
        
        assert user is not None
        assert user.user_id == sample_user.user_id
        assert user.username == sample_user.username
    
    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, auth_service):
        """Test verification of invalid token."""
        with pytest.raises(InvalidCredentialsError, match="令牌验证失败"):
            await auth_service.verify_token("invalid.token.here")
    
    @pytest.mark.asyncio
    async def test_verify_token_user_not_found(self, auth_service, mock_user_dao, jwt_manager):
        """Test verification when user doesn't exist."""
        token = jwt_manager.create_access_token(
            user_id="nonexistent-user",
            username="nonexistent"
        )
        
        mock_user_dao.get_by_id.return_value = None
        
        with pytest.raises(InvalidCredentialsError):
            await auth_service.verify_token(token)
    
    @pytest.mark.asyncio
    async def test_verify_token_inactive_user(self, auth_service, mock_user_dao, sample_user, jwt_manager):
        """Test verification with inactive user."""
        token = jwt_manager.create_access_token(
            user_id=sample_user.user_id,
            username=sample_user.username
        )
        
        sample_user.is_active = False
        mock_user_dao.get_by_id.return_value = sample_user
        
        with pytest.raises(InvalidCredentialsError, match="用户账号已被停用"):
            await auth_service.verify_token(token)


class TestTokenRefresh:
    """Tests for token refresh."""
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, auth_service, mock_user_dao, sample_user, jwt_manager):
        """Test successful token refresh."""
        old_token = jwt_manager.create_access_token(
            user_id=sample_user.user_id,
            username=sample_user.username
        )
        
        mock_user_dao.get_by_id.return_value = sample_user
        
        # Wait a moment to ensure different timestamps
        await asyncio.sleep(0.1)
        
        new_token = await auth_service.refresh_token(old_token)
        
        assert new_token is not None
        assert new_token != old_token
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, auth_service):
        """Test refreshing invalid token."""
        with pytest.raises(InvalidCredentialsError, match="令牌刷新失败"):
            await auth_service.refresh_token("invalid.token.here")


class TestLogout:
    """Tests for logout functionality."""
    
    @pytest.mark.asyncio
    async def test_logout_success(self, auth_service, mock_user_dao, sample_user, jwt_manager):
        """Test successful logout."""
        token = jwt_manager.create_access_token(
            user_id=sample_user.user_id,
            username=sample_user.username
        )
        
        mock_user_dao.get_by_id.return_value = sample_user
        
        result = await auth_service.logout(token)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_logout_invalid_token(self, auth_service):
        """Test logout with invalid token."""
        result = await auth_service.logout("invalid.token.here")
        
        assert result is False


class TestChangePassword:
    """Tests for password change."""
    
    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_service, mock_user_dao, sample_user):
        """Test successful password change."""
        mock_user_dao.get_by_id.return_value = sample_user
        mock_user_dao.update.return_value = sample_user
        
        result = await auth_service.change_password(
            user_id=sample_user.user_id,
            old_password="Password123",
            new_password="NewPassword456"
        )
        
        assert result is True
        mock_user_dao.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_old_password(self, auth_service, mock_user_dao, sample_user):
        """Test password change with wrong old password."""
        mock_user_dao.get_by_id.return_value = sample_user
        
        with pytest.raises(InvalidCredentialsError, match="当前密码错误"):
            await auth_service.change_password(
                user_id=sample_user.user_id,
                old_password="WrongPassword",
                new_password="NewPassword456"
            )
    
    @pytest.mark.asyncio
    async def test_change_password_weak_new_password(self, auth_service, mock_user_dao, sample_user):
        """Test password change with weak new password."""
        mock_user_dao.get_by_id.return_value = sample_user
        
        with pytest.raises(WeakPasswordError):
            await auth_service.change_password(
                user_id=sample_user.user_id,
                old_password="Password123",
                new_password="weak"
            )
    
    @pytest.mark.asyncio
    async def test_change_password_user_not_found(self, auth_service, mock_user_dao):
        """Test password change for non-existent user."""
        mock_user_dao.get_by_id.return_value = None
        
        with pytest.raises(UserNotFoundError, match="用户不存在"):
            await auth_service.change_password(
                user_id="nonexistent",
                old_password="Password123",
                new_password="NewPassword456"
            )


class TestUserRetrieval:
    """Tests for user retrieval methods."""
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, auth_service, mock_user_dao, sample_user):
        """Test getting user by ID."""
        mock_user_dao.get_by_id.return_value = sample_user
        
        user = await auth_service.get_user_by_id(sample_user.user_id)
        
        assert user is not None
        assert user.user_id == sample_user.user_id
    
    @pytest.mark.asyncio
    async def test_get_user_by_username(self, auth_service, mock_user_dao, sample_user):
        """Test getting user by username."""
        mock_user_dao.get_by_username.return_value = sample_user
        
        user = await auth_service.get_user_by_username(sample_user.username)
        
        assert user is not None
        assert user.username == sample_user.username
