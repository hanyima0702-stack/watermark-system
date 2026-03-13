"""
Authentication and service dependencies for FastAPI routes.
"""

from typing import Optional
from fastapi import Request, HTTPException, status
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)


class ServiceClient:
    """Client for calling internal microservices."""
    
    def __init__(self):
        try:
            from .config import settings
        except ImportError:
            from gateway.api.config import settings
        
        self.business_url = settings.BUSINESS_SERVICE_URL
        self.storage_url = settings.STORAGE_SERVICE_URL
        self.engine_url = settings.ENGINE_SERVICE_URL
    
    async def call_business_service(self, method: str, path: str, **kwargs):
        """Call the business service."""
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method, f"{self.business_url}{path}", **kwargs
            )
            response.raise_for_status()
            return response.json()


async def get_service_client() -> ServiceClient:
    """
    Dependency function to get a service client for calling internal services.
    
    Returns:
        ServiceClient instance
    """
    return ServiceClient()


class AuthenticationRequired(Exception):
    """Exception raised when authentication is required but not provided."""
    pass


async def get_current_user(request: Request) -> dict:
    """
    Dependency function to get the current authenticated user.
    Verifies the JWT token directly from the Authorization header.
    """
    # First try request state (set by middleware if working)
    user_id = getattr(request.state, 'user_id', None) or request.scope.get('user_id')
    username = getattr(request.state, 'username', None) or request.scope.get('username')

    if user_id:
        user_info = {"user_id": user_id, "username": username}
        if hasattr(request.state, "token_exp"):
            user_info["token_exp"] = request.state.token_exp
        return user_info

    # Fallback: verify JWT directly in the dependency
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Authentication required but no token provided", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    token = auth_header.split(" ")[1]

    try:
        from gateway.api.routers.auth import _auth_service
        if not _auth_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        token_data = _auth_service.jwt_manager.verify_token(token)
        return {
            "user_id": token_data.user_id,
            "username": token_data.username,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Token verification failed", path=request.url.path, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


async def get_current_user_optional(request: Request) -> Optional[dict]:
    """
    Dependency function to get the current user if authenticated, None otherwise.
    
    This is useful for routes that work differently for authenticated vs
    unauthenticated users but don't strictly require authentication.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary containing user information if authenticated, None otherwise
        
    Example:
        @app.get("/optional-auth")
        async def optional_route(user: Optional[dict] = Depends(get_current_user_optional)):
            if user:
                return {"message": f"Hello {user['username']}"}
            return {"message": "Hello guest"}
    """
    if not hasattr(request.state, "user_id"):
        return None
    
    user_info = {
        "user_id": request.state.user_id,
        "username": request.state.username,
    }
    
    if hasattr(request.state, "token_exp"):
        user_info["token_exp"] = request.state.token_exp
    
    return user_info


def require_auth(func):
    """
    Decorator to require authentication for a route handler.
    
    This decorator can be used on async route handler functions to ensure
    the user is authenticated. It checks for user_id in request.state.
    
    Args:
        func: Async route handler function
        
    Returns:
        Wrapped function that checks authentication
        
    Raises:
        HTTPException: 401 if user is not authenticated
        
    Example:
        @app.get("/protected")
        @require_auth
        async def protected_route(request: Request):
            user_id = request.state.user_id
            return {"user_id": user_id}
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find the request object in args or kwargs
        request = None
        
        # Check args for Request object
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # Check kwargs for Request object
        if not request and 'request' in kwargs:
            request = kwargs['request']
        
        if not request:
            logger.error("Request object not found in route handler arguments")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        
        # Check if user is authenticated
        if not hasattr(request.state, "user_id"):
            logger.warning(
                "Authentication required but user not found in request state",
                path=request.url.path
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Call the original function
        return await func(*args, **kwargs)
    
    return wrapper


def get_user_id(request: Request) -> str:
    """
    Helper function to get user ID from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: 401 if user is not authenticated
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.user_id


def get_username(request: Request) -> str:
    """
    Helper function to get username from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Username string
        
    Raises:
        HTTPException: 401 if user is not authenticated
    """
    if not hasattr(request.state, "username"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return request.state.username
