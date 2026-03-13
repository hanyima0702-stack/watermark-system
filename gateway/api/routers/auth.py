"""
Authentication API Router
认证相关的API路由
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from fastapi.responses import JSONResponse

from gateway.api.models import (
    RegisterRequest,
    LoginRequest,
    LoginResponse,
    UserResponse,
    SessionResponse,
    ErrorResponse
)
from gateway.api.auth.auth_service import (
    AuthService,
    InvalidCredentialsError,
    UserAlreadyExistsError,
    UserNotFoundError,
    WeakPasswordError
)


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"]
)

# Global auth service instance (will be set during app startup)
_auth_service: Optional[AuthService] = None


def set_auth_service(auth_service: AuthService):
    """Set the global auth service instance."""
    global _auth_service
    _auth_service = auth_service


def get_auth_service() -> AuthService:
    """Dependency to get auth service."""
    if _auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="认证服务未初始化"
        )
    return _auth_service


@router.post(
    "/register",
    response_model=LoginResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="注册新用户账号"
)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    用户注册端点
    
    - **username**: 用户名（3-20个字符）
    - **password**: 密码（至少8位，包含字母和数字）
    - **email**: 邮箱地址
    
    返回JWT令牌和用户信息
    """
    try:
        # Register user
        user, token = await auth_service.register(
            username=request.username,
            password=request.password,
            email=request.email
        )
        
        # Build response
        user_response = UserResponse(
            id=user.user_id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.updated_at,
            is_active=user.is_active
        )
        
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=86400,  # 24 hours
            user=user_response
        )
        
    except UserAlreadyExistsError as e:
        logger.warning(f"Registration failed - user exists: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    
    except WeakPasswordError as e:
        logger.warning(f"Registration failed - weak password: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后重试"
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="用户登录",
    description="使用用户名和密码登录"
)
async def login(
    request: LoginRequest,
    raw_request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    用户登录端点
    
    - **username**: 用户名
    - **password**: 密码
    
    返回JWT令牌和用户信息
    """
    try:
        # Extract client info
        ip_address = raw_request.client.host if raw_request.client else None
        user_agent = raw_request.headers.get("user-agent")
        
        # Authenticate user
        user, token = await auth_service.login(
            username=request.username,
            password=request.password,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Build response
        user_response = UserResponse(
            id=user.user_id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.updated_at,
            is_active=user.is_active
        )
        
        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=86400,  # 24 hours
            user=user_response
        )
        
    except (InvalidCredentialsError, UserNotFoundError) as e:
        logger.warning(f"Login failed for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )
    
    except Exception as e:
        logger.error(f"Login failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录失败，请稍后重试"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="用户登出",
    description="登出当前用户"
)
async def logout(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    用户登出端点
    
    需要在请求头中提供JWT令牌：
    Authorization: Bearer <token>
    """
    try:
        # Extract token from header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供认证令牌"
            )
        
        token = authorization.split(" ")[1]
        
        # Logout user
        success = await auth_service.logout(token)
        
        if success:
            return {"message": "登出成功"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="登出失败"
            )
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出失败，请稍后重试"
        )


@router.post(
    "/refresh",
    response_model=LoginResponse,
    summary="刷新令牌",
    description="使用现有令牌获取新令牌"
)
async def refresh_token(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    刷新JWT令牌端点
    
    需要在请求头中提供JWT令牌：
    Authorization: Bearer <token>
    
    返回新的JWT令牌和用户信息
    """
    try:
        # Extract token from header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供认证令牌"
            )
        
        token = authorization.split(" ")[1]
        
        # Verify old token and get user
        user = await auth_service.verify_token(token)
        
        # Generate new token
        new_token = await auth_service.refresh_token(token)
        
        # Build response
        user_response = UserResponse(
            id=user.user_id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.updated_at,
            is_active=user.is_active
        )
        
        return LoginResponse(
            access_token=new_token,
            token_type="bearer",
            expires_in=86400,  # 24 hours
            user=user_response
        )
        
    except InvalidCredentialsError as e:
        logger.warning(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期"
        )
    
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="令牌刷新失败，请稍后重试"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
    description="获取当前登录用户的信息"
)
async def get_current_user(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    获取当前用户信息端点
    
    需要在请求头中提供JWT令牌：
    Authorization: Bearer <token>
    
    返回当前用户的详细信息
    """
    try:
        # Extract token from header
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供认证令牌"
            )
        
        token = authorization.split(" ")[1]
        
        # Verify token and get user
        user = await auth_service.verify_token(token)
        
        # Build response
        return UserResponse(
            id=user.user_id,
            username=user.username,
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.updated_at,
            is_active=user.is_active
        )
        
    except InvalidCredentialsError as e:
        logger.warning(f"Get current user failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期"
        )
    
    except Exception as e:
        logger.error(f"Get current user failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败，请稍后重试"
        )


@router.get(
    "/sessions",
    response_model=List[SessionResponse],
    summary="获取活跃会话列表",
    description="获取当前用户的所有活跃会话"
)
async def get_sessions(
    authorization: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    获取当前用户的活跃会话列表

    需要在请求头中提供JWT令牌：
    Authorization: Bearer <token>
    """
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未提供认证令牌"
            )

        token = authorization.split(" ")[1]
        user = await auth_service.verify_token(token)

        sessions = await auth_service.get_active_sessions(user.user_id)

        return [
            SessionResponse(
                id=s.id,
                created_at=s.created_at,
                expires_at=s.expires_at,
                last_activity_at=s.last_activity_at,
                ip_address=s.ip_address,
                user_agent=s.user_agent,
                is_active=s.is_active
            )
            for s in sessions
        ]

    except InvalidCredentialsError as e:
        logger.warning(f"Get sessions failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Get sessions failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取会话列表失败，请稍后重试"
        )
