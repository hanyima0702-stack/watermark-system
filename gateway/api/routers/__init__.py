"""
API Routers
API路由模块，支持版本管理
"""

from .v1 import router as v1_router
from .v2 import router as v2_router

__all__ = ["v1_router", "v2_router"]
