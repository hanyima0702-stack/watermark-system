"""
API Gateway Main Application
统一API入口，支持请求路由、负载均衡、限流、监控和版本管理
"""

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import structlog
from prometheus_client import make_asgi_app

try:
    # 尝试相对导入（作为包运行时）
    from .middleware import (
        RateLimitMiddleware,
        LoggingMiddleware,
        AuthenticationMiddleware,
        MetricsMiddleware
    )
    from .routers import v1_router, v2_router
    from .routers.auth import router as auth_router, set_auth_service
    from .config import settings
    from .health import health_router
    from .auth.auth_service import AuthService
    from .auth.jwt_manager import JWTManager
except ImportError:
    # 绝对导入（直接运行时）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from gateway.api.middleware import (
        RateLimitMiddleware,
        LoggingMiddleware,
        AuthenticationMiddleware,
        MetricsMiddleware
    )
    from gateway.api.routers import v1_router, v2_router
    from gateway.api.routers.auth import router as auth_router, set_auth_service
    from gateway.api.config import settings
    from gateway.api.health import health_router
    from gateway.api.auth.auth_service import AuthService
    from gateway.api.auth.jwt_manager import JWTManager


logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting API Gateway", version=settings.API_VERSION)
    
    # Initialize auth service
    try:
        from storage.database_manager import DatabaseManager, init_database
        from shared.config import DatabaseConfig
        
        # Create database config
        db_config = DatabaseConfig()
        
        # Initialize global database manager (used by file_service etc.)
        db_manager = await init_database(db_config.connection_url)
        
        # Create JWT manager
        jwt_manager = JWTManager(
            secret_key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
            expire_minutes=settings.JWT_EXPIRE_MINUTES
        )
        
        # Create and initialize auth service (reuses the same db_manager)
        auth_service = AuthService(db_manager, jwt_manager)
        await auth_service.initialize()
        
        # Set global auth service
        set_auth_service(auth_service)
        
        logger.info("Auth service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize auth service: {str(e)}", exc_info=True)
    
    yield
    
    logger.info("Shutting down API Gateway")


# 创建FastAPI应用
app = FastAPI(
    title="Watermark System API Gateway",
    description="明暗水印统一嵌入与识别系统 API 网关",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 自定义中间件
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)

# 挂载Prometheus metrics端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# 注册路由
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(auth_router, tags=["Authentication"])
app.include_router(v1_router, prefix="/api/v1", tags=["API v1"])
app.include_router(v2_router, prefix="/api/v2", tags=["API v2"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id if hasattr(request.state, "request_id") else None
        }
    )


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Watermark System API Gateway",
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
