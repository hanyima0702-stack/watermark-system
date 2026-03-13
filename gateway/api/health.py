"""
Health Check Endpoints
健康检查和服务状态监控
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Optional
import time
from datetime import datetime

try:
    from .config import settings
except ImportError:
    from gateway.api.config import settings


# 创建健康检查路由器
health_router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, str]


class ReadinessResponse(BaseModel):
    """就绪检查响应"""
    ready: bool
    services: Dict[str, bool]
    message: Optional[str] = None


# 记录启动时间
START_TIME = time.time()


@health_router.get("/", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    用于Kubernetes liveness probe
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.API_VERSION,
        uptime=time.time() - START_TIME,
        services={
            "api_gateway": "up",
            "business_service": "unknown",
            "storage_service": "unknown",
            "engine_service": "unknown"
        }
    )


@health_router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    就绪检查端点
    用于Kubernetes readiness probe
    检查所有依赖服务是否就绪
    """
    # TODO: 实际检查各个服务的连接状态
    services_ready = {
        "redis": True,  # 检查Redis连接
        "database": True,  # 检查数据库连接
        "business_service": True,  # 检查业务服务
        "storage_service": True,  # 检查存储服务
    }
    
    all_ready = all(services_ready.values())
    
    return ReadinessResponse(
        ready=all_ready,
        services=services_ready,
        message="All services ready" if all_ready else "Some services not ready"
    )


@health_router.get("/live")
async def liveness_check():
    """
    存活检查端点
    简单返回200表示服务存活
    """
    return {"status": "alive"}
