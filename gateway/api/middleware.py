"""
API Gateway Middleware
实现限流、日志、认证、监控等中间件
"""

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
import structlog
from typing import Callable
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

try:
    from .config import settings
except ImportError:
    from gateway.api.config import settings


logger = structlog.get_logger(__name__)

# Prometheus指标
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'api_active_requests',
    'Number of active requests'
)

RATE_LIMIT_EXCEEDED = Counter(
    'api_rate_limit_exceeded_total',
    'Total rate limit exceeded events',
    ['endpoint']
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # 记录请求完成
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=f"{duration:.3f}s"
            )
            
            # 添加请求ID到响应头
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration=f"{duration:.3f}s",
                exc_info=exc
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = None
        if settings.RATE_LIMIT_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
            except Exception as e:
                logger.warning("Failed to connect to Redis for rate limiting", error=str(e))
    
    async def dispatch(self, request: Request, call_next: Callable):
        if not settings.RATE_LIMIT_ENABLED or not self.redis_client:
            return await call_next(request)
        
        # 获取客户端标识（IP或用户ID）
        client_id = request.client.host if request.client else "unknown"
        if hasattr(request.state, "user_id"):
            client_id = request.state.user_id
        
        # 限流键
        rate_limit_key = f"rate_limit:{client_id}"
        
        try:
            # 获取当前请求计数
            current_requests = await self.redis_client.get(rate_limit_key)
            
            if current_requests and int(current_requests) >= settings.RATE_LIMIT_REQUESTS:
                RATE_LIMIT_EXCEEDED.labels(endpoint=request.url.path).inc()
                logger.warning(
                    "Rate limit exceeded",
                    client_id=client_id,
                    path=request.url.path
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Maximum {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds"
                    }
                )
            
            # 增加计数
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_limit_key)
            pipe.expire(rate_limit_key, settings.RATE_LIMIT_WINDOW)
            await pipe.execute()
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            # 限流失败时允许请求通过
        
        return await call_next(request)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    # 不需要认证的路径
    PUBLIC_PATHS = [
        "/", "/health", "/docs", "/redoc", "/openapi.json", "/metrics",
        "/api/v1/auth/register", "/api/v1/auth/login"
    ]
    
    def __init__(self, app: ASGIApp, jwt_manager=None):
        super().__init__(app)
        self.jwt_manager = jwt_manager

    def _get_jwt_manager(self):
        """Get jwt_manager lazily — falls back to the global auth service."""
        if self.jwt_manager:
            return self.jwt_manager
        try:
            from gateway.api.routers.auth import _auth_service
            if _auth_service:
                return _auth_service.jwt_manager
        except Exception:
            pass
        return None
    
    async def dispatch(self, request: Request, call_next: Callable):
        # 检查是否为公开路径
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # 获取Authorization头
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning(
                "Missing or invalid authorization header",
                path=request.url.path,
                client_ip=request.client.host if request.client else None
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Unauthorized",
                    "message": "Missing or invalid authorization header"
                }
            )
        
        # 提取JWT令牌
        token = auth_header.split(" ")[1]
        
        # 验证JWT令牌
        jwt_manager = self._get_jwt_manager()
        if not jwt_manager:
            logger.error("JWT manager not configured")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "message": "Authentication service not available"
                }
            )
        
        try:
            # 调用JWT管理器验证令牌
            token_data = jwt_manager.verify_token(token)
            
            # 将用户信息添加到request.state
            request.state.user_id = token_data.user_id
            request.state.username = token_data.username
            request.state.token_exp = token_data.exp
            # Also store in scope for compatibility with BaseHTTPMiddleware
            request.scope['user_id'] = token_data.user_id
            request.scope['username'] = token_data.username
            
            logger.debug(
                "Authentication successful",
                user_id=token_data.user_id,
                username=token_data.username,
                path=request.url.path
            )
            
        except Exception as e:
            # 处理令牌过期和无效情况
            error_message = str(e)
            
            if "expired" in error_message.lower():
                logger.warning(
                    "Token expired",
                    path=request.url.path,
                    error=error_message
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "TokenExpired",
                        "message": "Token has expired, please login again"
                    }
                )
            else:
                logger.warning(
                    "Token verification failed",
                    path=request.url.path,
                    error=error_message
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "InvalidToken",
                        "message": "Invalid token"
                    }
                )
        
        return await call_next(request)


class MetricsMiddleware(BaseHTTPMiddleware):
    """监控指标中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        if not settings.METRICS_ENABLED:
            return await call_next(request)
        
        # 增加活跃请求计数
        ACTIVE_REQUESTS.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # 记录请求指标
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        finally:
            # 减少活跃请求计数
            ACTIVE_REQUESTS.dec()
