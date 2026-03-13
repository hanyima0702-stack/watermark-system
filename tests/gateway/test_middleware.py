"""
Tests for authentication middleware and dependencies.

Note: These tests verify the authentication middleware and dependency functions.
Run with: python -m pytest tests/gateway/test_middleware.py -v

Requirements:
- FastAPI and dependencies must be installed
- Run from watermark-system directory
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

# Mark all tests to skip if dependencies not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("fastapi", reason="FastAPI not installed"),
    reason="Required dependencies not available"
)

from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.testclient import TestClient

from gateway.api.middleware import AuthenticationMiddleware
from gateway.api.dependencies import (
    get_current_user,
    get_current_user_optional,
    require_auth,
    get_user_id,
    get_username
)
from gateway.api.auth.jwt_manager import JWTManager, TokenData


@pytest.fixture
def jwt_manager():
    """Create a JWT manager for testing."""
    return JWTManager(
        secret_key="test-secret-key",
        algorithm="HS256",
        expire_minutes=30
    )


@pytest.fixture
def valid_token(jwt_manager):
    """Create a valid JWT token."""
    return jwt_manager.create_access_token(
        user_id="test-user-123",
        username="testuser"
    )


@pytest.fixture
def expired_token(jwt_manager):
    """Create an expired JWT token."""
    return jwt_manager.create_access_token(
        user_id="test-user-123",
        username="testuser",
        expires_delta=timedelta(seconds=-1)  # Already expired
    )


@pytest.fixture
def app_with_auth(jwt_manager):
    """Create a FastAPI app with authentication middleware."""
    app = FastAPI()
    
    # Add authentication middleware
    app.add_middleware(AuthenticationMiddleware, jwt_manager=jwt_manager)
    
    # Public route
    @app.get("/public")
    async def public_route():
        return {"message": "public"}
    
    # Health check (public)
    @app.get("/health")
    async def health_route():
        return {"status": "ok"}
    
    # Protected route
    @app.get("/protected")
    async def protected_route(request: Request):
        return {
            "message": "protected",
            "user_id": request.state.user_id,
            "username": request.state.username
        }
    
    # Route with dependency
    @app.get("/with-dependency")
    async def route_with_dependency(user: dict = Depends(get_current_user)):
        return {
            "message": "with dependency",
            "user": user
        }
    
    # Route with optional auth
    @app.get("/optional-auth")
    async def route_optional_auth(user: dict = Depends(get_current_user_optional)):
        if user:
            return {"message": f"Hello {user['username']}"}
        return {"message": "Hello guest"}
    
    # Route with decorator
    @app.get("/with-decorator")
    @require_auth
    async def route_with_decorator(request: Request):
        return {
            "message": "with decorator",
            "user_id": request.state.user_id
        }
    
    return app


class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware."""
    
    def test_public_route_no_auth_required(self, app_with_auth):
        """Test that public routes don't require authentication."""
        client = TestClient(app_with_auth)
        response = client.get("/public")
        
        assert response.status_code == 200
        assert response.json() == {"message": "public"}
    
    def test_health_check_no_auth_required(self, app_with_auth):
        """Test that health check doesn't require authentication."""
        client = TestClient(app_with_auth)
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_protected_route_with_valid_token(self, app_with_auth, valid_token):
        """Test protected route with valid JWT token."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "protected"
        assert data["user_id"] == "test-user-123"
        assert data["username"] == "testuser"
    
    def test_protected_route_without_token(self, app_with_auth):
        """Test protected route without token returns 401."""
        client = TestClient(app_with_auth)
        response = client.get("/protected")
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "Unauthorized"
        assert "Missing or invalid" in data["message"]
    
    def test_protected_route_with_invalid_token(self, app_with_auth):
        """Test protected route with invalid token returns 401."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer invalid-token-here"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "InvalidToken"
        assert "Invalid token" in data["message"]
    
    def test_protected_route_with_expired_token(self, app_with_auth, expired_token):
        """Test protected route with expired token returns 401."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {expired_token}"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "TokenExpired"
        assert "expired" in data["message"].lower()
    
    def test_protected_route_with_malformed_header(self, app_with_auth, valid_token):
        """Test protected route with malformed authorization header."""
        client = TestClient(app_with_auth)
        
        # Missing "Bearer" prefix
        response = client.get(
            "/protected",
            headers={"Authorization": valid_token}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "Unauthorized"
    
    def test_middleware_without_jwt_manager(self):
        """Test middleware without JWT manager configured."""
        app = FastAPI()
        app.add_middleware(AuthenticationMiddleware, jwt_manager=None)
        
        @app.get("/protected")
        async def protected():
            return {"message": "protected"}
        
        client = TestClient(app)
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer some-token"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""
    
    def test_get_current_user_with_valid_auth(self, app_with_auth, valid_token):
        """Test get_current_user with authenticated request."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/with-dependency",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "with dependency"
        assert data["user"]["user_id"] == "test-user-123"
        assert data["user"]["username"] == "testuser"
    
    def test_get_current_user_without_auth(self, app_with_auth):
        """Test get_current_user without authentication."""
        client = TestClient(app_with_auth)
        response = client.get("/with-dependency")
        
        assert response.status_code == 401


class TestGetCurrentUserOptional:
    """Tests for get_current_user_optional dependency."""
    
    def test_optional_auth_with_token(self, app_with_auth, valid_token):
        """Test optional auth with valid token."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/optional-auth",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello testuser"
    
    def test_optional_auth_without_token(self, app_with_auth):
        """Test optional auth without token."""
        client = TestClient(app_with_auth)
        response = client.get("/optional-auth")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Hello guest"


class TestRequireAuthDecorator:
    """Tests for require_auth decorator."""
    
    def test_decorator_with_valid_auth(self, app_with_auth, valid_token):
        """Test decorator with authenticated request."""
        client = TestClient(app_with_auth)
        response = client.get(
            "/with-decorator",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "with decorator"
        assert data["user_id"] == "test-user-123"
    
    def test_decorator_without_auth(self, app_with_auth):
        """Test decorator without authentication."""
        client = TestClient(app_with_auth)
        response = client.get("/with-decorator")
        
        assert response.status_code == 401


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_user_id(self):
        """Test get_user_id helper function."""
        # Create mock request with user_id
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.user_id = "test-user-123"
        
        user_id = get_user_id(request)
        assert user_id == "test-user-123"
    
    def test_get_user_id_without_auth(self):
        """Test get_user_id without authentication."""
        # Create mock request without user_id
        request = Mock(spec=Request)
        request.state = Mock(spec=[])  # Empty state
        
        with pytest.raises(HTTPException) as exc_info:
            get_user_id(request)
        
        assert exc_info.value.status_code == 401
    
    def test_get_username(self):
        """Test get_username helper function."""
        # Create mock request with username
        request = Mock(spec=Request)
        request.state = Mock()
        request.state.username = "testuser"
        
        username = get_username(request)
        assert username == "testuser"
    
    def test_get_username_without_auth(self):
        """Test get_username without authentication."""
        # Create mock request without username
        request = Mock(spec=Request)
        request.state = Mock(spec=[])  # Empty state
        
        with pytest.raises(HTTPException) as exc_info:
            get_username(request)
        
        assert exc_info.value.status_code == 401


class TestMiddlewareIntegration:
    """Integration tests for middleware with various scenarios."""
    
    def test_multiple_requests_with_same_token(self, app_with_auth, valid_token):
        """Test multiple requests with the same valid token."""
        client = TestClient(app_with_auth)
        
        for _ in range(3):
            response = client.get(
                "/protected",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            assert response.status_code == 200
    
    def test_request_state_isolation(self, app_with_auth, jwt_manager):
        """Test that request state is isolated between requests."""
        client = TestClient(app_with_auth)
        
        # Create two different tokens
        token1 = jwt_manager.create_access_token("user1", "username1")
        token2 = jwt_manager.create_access_token("user2", "username2")
        
        # Make request with token1
        response1 = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token1}"}
        )
        assert response1.json()["user_id"] == "user1"
        
        # Make request with token2
        response2 = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token2}"}
        )
        assert response2.json()["user_id"] == "user2"
    
    def test_public_paths_bypass_auth(self, app_with_auth):
        """Test that public paths defined in middleware are accessible."""
        client = TestClient(app_with_auth)
        
        # These should all be accessible without auth
        public_paths = ["/public", "/health"]
        
        for path in public_paths:
            response = client.get(path)
            assert response.status_code == 200  # Should be accessible
