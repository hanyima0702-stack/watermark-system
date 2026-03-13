"""
API v2 Router
第二版API路由定义（向后兼容v1，添加新特性）
"""

from fastapi import APIRouter, Depends
from typing import List

from .v1 import router as v1_router
from ..dependencies import get_current_user, get_service_client


router = APIRouter()

# 继承v1的所有路由
router.include_router(v1_router)


# ============= v2新增功能 =============

@router.post("/watermark/batch")
async def batch_watermark_tasks(
    file_ids: List[str],
    config_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    批量水印任务
    v2新增功能：支持批量处理多个文件
    """
    # TODO: 实现批量任务创建
    return {
        "batch_id": "batch_123",
        "task_ids": [],
        "message": "Batch tasks created successfully"
    }


@router.get("/watermark/templates")
async def list_watermark_templates(
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    列出水印模板
    v2新增功能：预定义的水印模板
    """
    # TODO: 实现模板列表
    return []


@router.post("/watermark/preview")
async def preview_watermark(
    file_id: str,
    config_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    水印预览
    v2新增功能：在实际嵌入前预览水印效果
    """
    # TODO: 实现水印预览
    return {
        "preview_url": "",
        "expires_at": None
    }
