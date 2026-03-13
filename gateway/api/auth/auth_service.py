"""User authentication service."""

import uuid
import hashlib
import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta

from .password_utils import (
    hash_password,
    verify_password,
    validate_password_strength
)
from .jwt_manager import JWTManager
from storage.database_manager import DatabaseManager
from storage.models.user import User
from storage.models.user_session import UserSession


logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Exception raised when credentials are invalid."""
    pass


class UserAlreadyExistsError(AuthenticationError):
    """Exception raised when user already exists."""
    pass


class UserNotFoundError(AuthenticationError):
    """Exception raised when user is not found."""
    pass


class WeakPasswordError(AuthenticationError):
    """Exception raised when password is too weak."""
    pass


class AuthService:
    """Service for handling user authentication and authorization."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        jwt_manager: JWTManager
    ):
        """
        Initialize authentication service.
        
        Args:
            db_manager: Database manager instance
            jwt_manager: JWT manager instance
        """
        self.db_manager = db_manager
        self.jwt_manager = jwt_manager
        self.user_dao = None
        self.session_dao = None
    
    async def initialize(self):
        """Initialize the auth service."""
        # Get UserDAO from database manager
        self.user_dao = self.db_manager.get_dao('user')
        self.session_dao = self.db_manager.get_dao('user_session')
        logger.info("AuthService initialized")
    
    async def register(
        self,
        username: str,
        password: str,
        email: str,
        department: Optional[str] = None
    ) -> Tuple[User, str]:
        """
        Register a new user.
        
        Args:
            username: Username for the new user
            password: Plain text password
            email: User email address
            department: Optional department name
            
        Returns:
            Tuple of (User object, JWT token)
            
        Raises:
            UserAlreadyExistsError: If username or email already exists
            WeakPasswordError: If password doesn't meet strength requirements
        """
        # Validate password strength
        is_valid, error_message = validate_password_strength(password)
        if not is_valid:
            raise WeakPasswordError(error_message)
        
        # Check if username already exists
        existing_user = await self.user_dao.get_by_username(username)
        if existing_user:
            raise UserAlreadyExistsError("用户名已被使用")
        
        # Check if email already exists
        existing_email = await self.user_dao.get_by_email(email)
        if existing_email:
            raise UserAlreadyExistsError("邮箱已被使用")
        
        # Hash password
        password_hash = hash_password(password)
        
        # Create new user
        user_id = str(uuid.uuid4())
        new_user = User(
            user_id=user_id,
            username=username,
            email=email,
            department=department or "未分配",
            password_hash=password_hash,
            salt="",  # bcrypt includes salt in hash
            roles=["user"],  # Default role
            is_active=True
        )
        
        # Save to database
        created_user = await self.user_dao.create(new_user)
        
        # Generate JWT token
        token = self.jwt_manager.create_access_token(
            user_id=created_user.user_id,
            username=created_user.username
        )
        
        logger.info(f"User registered successfully: {username}")
        
        return created_user, token
    
    async def login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[User, str]:
        """
        Authenticate user and generate JWT token.
        
        Args:
            username: Username
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client User-Agent string
            
        Returns:
            Tuple of (User object, JWT token)
            
        Raises:
            InvalidCredentialsError: If credentials are invalid
            UserNotFoundError: If user doesn't exist
        """
        # Get user by username
        user = await self.user_dao.get_by_username(username)
        
        if not user:
            raise UserNotFoundError("用户名或密码错误")
        
        # Check if user is active
        if not user.is_active:
            raise InvalidCredentialsError("用户账号已被停用")
        
        # Verify password
        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError("用户名或密码错误")
        
        # Update last login time
        await self.user_dao.update(user.user_id, {
            'updated_at': datetime.utcnow()
        })
        
        # Generate JWT token
        token = self.jwt_manager.create_access_token(
            user_id=user.user_id,
            username=user.username
        )
        
        # Record session
        await self._create_session(
            user_id=user.user_id,
            token=token,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        logger.info(f"User logged in successfully: {username}")
        
        return user, token
    
    async def _create_session(
        self,
        user_id: str,
        token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create a session record in the database."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.utcnow() + timedelta(minutes=self.jwt_manager.expire_minutes)
        
        session = UserSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            last_activity_at=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        return await self.session_dao.create(session)
    
    async def verify_token(self, token: str) -> User:
        """
        Verify JWT token and return user.
        
        Args:
            token: JWT token string
            
        Returns:
            User object
            
        Raises:
            InvalidCredentialsError: If token is invalid
            UserNotFoundError: If user doesn't exist
        """
        try:
            # Verify and decode token
            token_data = self.jwt_manager.verify_token(token)
            
            # Get user from database
            user = await self.user_dao.get_by_id(token_data.user_id)
            
            if not user:
                raise UserNotFoundError("用户不存在")
            
            if not user.is_active:
                raise InvalidCredentialsError("用户账号已被停用")
            
            return user
            
        except Exception as e:
            logger.warning(f"Token verification failed: {str(e)}")
            raise InvalidCredentialsError("令牌验证失败")
    
    async def refresh_token(self, token: str) -> str:
        """
        Refresh an existing JWT token.
        
        Args:
            token: Existing JWT token
            
        Returns:
            New JWT token string
            
        Raises:
            InvalidCredentialsError: If token is invalid
        """
        try:
            # Verify the old token first
            user = await self.verify_token(token)
            
            # Generate new token
            new_token = self.jwt_manager.create_access_token(
                user_id=user.user_id,
                username=user.username
            )
            
            logger.info(f"Token refreshed for user: {user.username}")
            
            return new_token
            
        except Exception as e:
            logger.warning(f"Token refresh failed: {str(e)}")
            raise InvalidCredentialsError("令牌刷新失败")
    
    async def logout(self, token: str) -> bool:
        """
        Logout user (invalidate session).
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if logout successful
        """
        try:
            # Verify token is valid
            user = await self.verify_token(token)
            
            # Deactivate the session by token hash
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            await self.session_dao.deactivate_by_token_hash(token_hash)
            
            logger.info(f"User logged out: {user.username}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Logout failed: {str(e)}")
            return False
    
    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
            
        Raises:
            InvalidCredentialsError: If old password is incorrect
            WeakPasswordError: If new password is too weak
            UserNotFoundError: If user doesn't exist
        """
        # Get user
        user = await self.user_dao.get_by_id(user_id)
        if not user:
            raise UserNotFoundError("用户不存在")
        
        # Verify old password
        if not verify_password(old_password, user.password_hash):
            raise InvalidCredentialsError("当前密码错误")
        
        # Validate new password strength
        is_valid, error_message = validate_password_strength(new_password)
        if not is_valid:
            raise WeakPasswordError(error_message)
        
        # Hash new password
        new_password_hash = hash_password(new_password)
        
        # Update password
        result = await self.user_dao.update(user_id, {
            'password_hash': new_password_hash,
            'updated_at': datetime.utcnow()
        })
        
        if result:
            logger.info(f"Password changed for user: {user.username}")
            return True
        
        return False
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None
        """
        return await self.user_dao.get_by_id(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            User object or None
        """
        return await self.user_dao.get_by_username(username)
    
    async def get_active_sessions(self, user_id: str) -> list:
        """
        Get active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of active UserSession objects
        """
        return await self.session_dao.get_active_sessions_by_user(user_id)
