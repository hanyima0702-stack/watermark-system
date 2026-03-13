"""Unit tests for JWT manager."""

import pytest
from datetime import datetime, timedelta
from jose import JWTError
from gateway.api.auth.jwt_manager import JWTManager, TokenData


class TestJWTManager:
    """Tests for JWTManager class."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager instance for testing."""
        return JWTManager(
            secret_key="test-secret-key-12345",
            algorithm="HS256",
            expire_minutes=30
        )
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing."""
        return {
            "user_id": "user-123",
            "username": "testuser"
        }
    
    def test_create_access_token(self, jwt_manager, sample_user_data):
        """Test creating an access token."""
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_custom_expiry(self, jwt_manager, sample_user_data):
        """Test creating token with custom expiration time."""
        custom_delta = timedelta(minutes=60)
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"],
            expires_delta=custom_delta
        )
        
        assert token is not None
        
        # Verify the token has the custom expiration
        token_data = jwt_manager.verify_token(token)
        assert token_data.exp is not None
    
    def test_verify_token_success(self, jwt_manager, sample_user_data):
        """Test successful token verification."""
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        token_data = jwt_manager.verify_token(token)
        
        assert token_data.user_id == sample_user_data["user_id"]
        assert token_data.username == sample_user_data["username"]
        assert token_data.exp is not None
    
    def test_verify_token_invalid(self, jwt_manager):
        """Test verification of invalid token."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(JWTError):
            jwt_manager.verify_token(invalid_token)
    
    def test_verify_token_wrong_secret(self, sample_user_data):
        """Test verification with wrong secret key."""
        manager1 = JWTManager(secret_key="secret1")
        manager2 = JWTManager(secret_key="secret2")
        
        token = manager1.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        with pytest.raises(JWTError):
            manager2.verify_token(token)
    
    def test_decode_token(self, jwt_manager, sample_user_data):
        """Test decoding token without verification."""
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        payload = jwt_manager.decode_token(token)
        
        assert payload["user_id"] == sample_user_data["user_id"]
        assert payload["username"] == sample_user_data["username"]
        assert "exp" in payload
    
    def test_is_token_expired_valid(self, jwt_manager, sample_user_data):
        """Test checking if valid token is not expired."""
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        assert jwt_manager.is_token_expired(token) is False
    
    def test_is_token_expired_expired(self, jwt_manager, sample_user_data):
        """Test checking if expired token is detected."""
        # Create token that expires immediately
        token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"],
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        assert jwt_manager.is_token_expired(token) is True
    
    def test_is_token_expired_invalid(self, jwt_manager):
        """Test checking expiration of invalid token."""
        invalid_token = "invalid.token.here"
        
        assert jwt_manager.is_token_expired(invalid_token) is True
    
    def test_refresh_token(self, jwt_manager, sample_user_data):
        """Test refreshing a token."""
        original_token = jwt_manager.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        
        # Wait a moment to ensure different timestamps
        import time
        time.sleep(0.1)
        
        new_token = jwt_manager.refresh_token(original_token)
        
        assert new_token is not None
        assert new_token != original_token
        
        # Verify new token has same user data
        new_token_data = jwt_manager.verify_token(new_token)
        assert new_token_data.user_id == sample_user_data["user_id"]
        assert new_token_data.username == sample_user_data["username"]
    
    def test_refresh_token_invalid(self, jwt_manager):
        """Test refreshing an invalid token."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(JWTError):
            jwt_manager.refresh_token(invalid_token)
    
    def test_token_data_model(self):
        """Test TokenData model."""
        token_data = TokenData(
            user_id="user-123",
            username="testuser",
            exp=datetime.utcnow()
        )
        
        assert token_data.user_id == "user-123"
        assert token_data.username == "testuser"
        assert token_data.exp is not None
    
    def test_token_data_model_without_exp(self):
        """Test TokenData model without expiration."""
        token_data = TokenData(
            user_id="user-123",
            username="testuser"
        )
        
        assert token_data.user_id == "user-123"
        assert token_data.username == "testuser"
        assert token_data.exp is None
    
    def test_different_algorithms(self, sample_user_data):
        """Test JWT manager with different algorithms."""
        # Test with HS256
        manager_hs256 = JWTManager(
            secret_key="test-secret",
            algorithm="HS256"
        )
        token_hs256 = manager_hs256.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        assert manager_hs256.verify_token(token_hs256) is not None
        
        # Test with HS512
        manager_hs512 = JWTManager(
            secret_key="test-secret",
            algorithm="HS512"
        )
        token_hs512 = manager_hs512.create_access_token(
            user_id=sample_user_data["user_id"],
            username=sample_user_data["username"]
        )
        assert manager_hs512.verify_token(token_hs512) is not None
    
    def test_token_missing_fields(self, jwt_manager):
        """Test verification of token with missing required fields."""
        # Manually create a token with missing fields
        from jose import jwt
        
        payload = {"exp": datetime.utcnow() + timedelta(minutes=30)}
        token = jwt.encode(
            payload,
            jwt_manager.secret_key,
            algorithm=jwt_manager.algorithm
        )
        
        with pytest.raises(JWTError, match="missing required fields"):
            jwt_manager.verify_token(token)
