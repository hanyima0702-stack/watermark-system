"""
API v1 Router
第一版API路由定义
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from ..dependencies import get_current_user, get_service_client
from ..models import (
    WatermarkTaskRequest,
    WatermarkTaskResponse,
    TaskStatusResponse,
    FileUploadResponse,
    ExtractionRequest,
    ExtractionResponse,
    ConfigRequest,
    ConfigResponse,
    ChunkedUploadRequest,
    ChunkedUploadResponse,
    UploadStatusResponse,
    DownloadTokenResponse
)
from ..file_service import file_service


router = APIRouter()


# ============= 文件管理 =============

@router.post("/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    上传文件（小文件直接上传）
    对于大文件，请使用分片上传接口
    """
    result = await file_service.upload_file(
        file=file,
        user_id=current_user["user_id"],
        validate=True
    )
    
    return FileUploadResponse(
        file_id=result["file_id"],
        filename=result["filename"],
        file_size=result["file_size"],
        upload_time=datetime.fromisoformat(result["uploaded_at"]),
        status="uploaded"
    )


@router.post("/files/upload/chunked/init", response_model=ChunkedUploadResponse)
async def init_chunked_upload(
    request: ChunkedUploadRequest,
    current_user = Depends(get_current_user)
):
    """
    初始化分片上传
    返回upload_id用于后续分片上传
    """
    upload_id = file_service.upload_manager.create_upload_session(
        filename=request.filename,
        file_size=request.file_size,
        chunk_size=request.chunk_size,
        user_id=current_user["user_id"]
    )
    
    return ChunkedUploadResponse(
        upload_id=upload_id,
        chunk_size=request.chunk_size,
        total_chunks=(request.file_size + request.chunk_size - 1) // request.chunk_size
    )


@router.post("/files/upload/chunked/{upload_id}/chunk/{chunk_index}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    上传文件分片
    支持断点续传
    """
    chunk_data = await file.read()
    
    result = await file_service.upload_manager.upload_chunk(
        upload_id=upload_id,
        chunk_index=chunk_index,
        chunk_data=chunk_data
    )
    
    return result


@router.post("/files/upload/chunked/{upload_id}/complete")
async def complete_chunked_upload(
    upload_id: str,
    current_user = Depends(get_current_user)
):
    """
    完成分片上传
    合并所有分片为最终文件
    """
    file_path = await file_service.upload_manager.complete_upload(upload_id)
    
    # 计算文件哈希
    file_hash = await file_service.calculate_file_hash(file_path)
    
    return {
        "upload_id": upload_id,
        "file_path": file_path,
        "file_hash": file_hash,
        "status": "completed"
    }


@router.get("/files/upload/chunked/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    upload_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取分片上传状态
    用于断点续传时查询已上传的分片
    """
    status_data = file_service.upload_manager.get_upload_status(upload_id)
    
    return UploadStatusResponse(**status_data)


@router.delete("/files/upload/chunked/{upload_id}")
async def cancel_chunked_upload(
    upload_id: str,
    current_user = Depends(get_current_user)
):
    """取消分片上传"""
    success = await file_service.upload_manager.cancel_upload(upload_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload session not found"
        )
    
    return {"message": "Upload cancelled successfully"}


@router.get("/files/{file_id}")
async def download_file(
    file_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    下载文件
    验证用户权限并从MinIO获取文件
    """
    from fastapi.responses import StreamingResponse
    import io
    
    # 下载文件
    result = await file_service.download_file_by_id(
        file_id=file_id,
        user_id=current_user["user_id"],
        generate_presigned_url=False
    )
    
    # 返回文件流
    file_stream = io.BytesIO(result["file_data"])
    
    return StreamingResponse(
        file_stream,
        media_type=result["content_type"],
        headers={
            "Content-Disposition": f'attachment; filename="{result["filename"]}"',
            "Content-Length": str(result["file_size"])
        }
    )


@router.get("/files/{file_id}/download")
async def get_download_url(
    file_id: str,
    current_user = Depends(get_current_user)
):
    """
    获取文件下载URL
    生成预签名URL用于直接下载
    """
    result = await file_service.download_file_by_id(
        file_id=file_id,
        user_id=current_user["user_id"],
        generate_presigned_url=True
    )
    
    return {
        "file_id": result["file_id"],
        "filename": result["filename"],
        "download_url": result["download_url"],
        "expires_in": result["expires_in"]
    }


@router.post("/files/{file_id}/download-token", response_model=DownloadTokenResponse)
async def generate_download_token(
    file_id: str,
    expires_in: int = 3600,
    current_user = Depends(get_current_user)
):
    """
    生成下载令牌
    实现访问控制和临时下载链接
    """
    token = file_service.generate_download_token(
        file_id=file_id,
        user_id=current_user["user_id"],
        expires_in=expires_in
    )
    
    return DownloadTokenResponse(
        token=token,
        file_id=file_id,
        expires_in=expires_in,
        download_url=f"/api/v1/files/{file_id}?token={token}"
    )


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    删除文件
    验证用户权限，从MinIO删除文件，从MySQL删除元数据
    """
    success = await file_service.delete_file_by_id(
        file_id=file_id,
        user_id=current_user["user_id"]
    )
    
    if success:
        return {"message": "File deleted successfully", "file_id": file_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )


# ============= 水印任务 =============

@router.post("/watermark/embed", response_model=WatermarkTaskResponse)
async def create_watermark_task(
    request: WatermarkTaskRequest,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    创建水印嵌入任务
    支持明水印、暗水印或两者同时嵌入
    需要用户认证，任务关联到当前用户
    """
    user_id = current_user["user_id"]
    
    # TODO: 调用业务服务创建任务，传入user_id关联
    # task = await service_client.call_business_service(
    #     "POST", "/watermark/tasks",
    #     json={**request.dict(), "user_id": user_id}
    # )
    
    return WatermarkTaskResponse(
        task_id="task_123",
        status="pending",
        created_at=datetime.utcnow(),
        message=f"Task created successfully for user {user_id}"
    )


@router.get("/watermark/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """获取任务状态和进度（仅限当前用户的任务）"""
    user_id = current_user["user_id"]
    
    # TODO: 调用业务服务查询任务状态，验证任务属于当前用户
    return TaskStatusResponse(
        task_id=task_id,
        status="processing",
        progress=0.5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@router.post("/watermark/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """取消任务（仅限当前用户的任务）"""
    user_id = current_user["user_id"]
    
    # TODO: 调用业务服务取消任务，验证任务属于当前用户
    return {"message": "Task cancelled successfully"}


@router.get("/watermark/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """列出当前用户的水印任务"""
    user_id = current_user["user_id"]
    
    # TODO: 调用业务服务查询当前用户的任务列表
    return []


# ============= 水印提取 =============

@router.post("/watermark/extract", response_model=ExtractionResponse)
async def extract_watermark(
    request: ExtractionRequest,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    提取水印信息
    支持盲检测提取
    需要用户认证，验证文件属于当前用户
    """
    user_id = current_user["user_id"]
    
    # TODO: 调用提取服务，验证file_id属于当前用户
    return ExtractionResponse(
        result_id="result_123",
        file_id=request.file_id,
        extracted_data={},
        confidence_score=0.0,
        extraction_time=datetime.utcnow()
    )


# ============= 配置管理 =============

@router.post("/configs", response_model=ConfigResponse)
async def create_config(
    request: ConfigRequest,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """创建水印配置（关联到当前用户）"""
    import uuid
    from gateway.api.routers.auth import _auth_service

    user_id = current_user["user_id"]
    config_id = str(uuid.uuid4())[:8]

    if not _auth_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    config_dao = _auth_service.db_manager.get_dao('watermark_config')
    from storage.models.watermark_config import WatermarkConfig as WCModel
    new_cfg = WCModel(
        config_id=config_id,
        config_name=request.config_name,
        watermark_type=request.watermark_type.value if hasattr(request.watermark_type, 'value') else str(request.watermark_type),
        visible_config=request.visible_config or {},
        invisible_config=request.invisible_config or {},
        template_variables=request.template_variables or {},
        is_active=True,
        created_by=user_id,
    )
    await config_dao.create(new_cfg)

    return ConfigResponse(
        config_id=config_id,
        config_name=request.config_name,
        watermark_type=request.watermark_type,
        created_at=datetime.utcnow(),
    )


@router.get("/configs/{config_id}", response_model=ConfigResponse)
async def get_config(
    config_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """获取配置详情"""
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    config_dao = _auth_service.db_manager.get_dao('watermark_config')
    cfg = await config_dao.get_by_id(config_id)
    if not cfg or cfg.created_by != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Config not found")

    return ConfigResponse(
        config_id=cfg.config_id,
        config_name=cfg.config_name,
        watermark_type=cfg.watermark_type,
        created_at=cfg.created_at,
        is_active=cfg.is_active,
        visible_config=cfg.visible_config,
        invisible_config=cfg.invisible_config,
        template_variables=cfg.template_variables,
    )


@router.get("/configs", response_model=List[ConfigResponse])
async def list_configs(
    media_type: Optional[str] = None,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """列出用户的配置，可按 media_type 过滤"""
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        return []

    config_dao = _auth_service.db_manager.get_dao('watermark_config')
    configs = await config_dao.get_user_configs(current_user["user_id"])

    if media_type:
        configs = [c for c in configs if (c.template_variables or {}).get('media_type') == media_type]

    return [
        ConfigResponse(
            config_id=c.config_id,
            config_name=c.config_name,
            watermark_type=c.watermark_type,
            created_at=c.created_at,
            is_active=c.is_active,
        )
        for c in configs
    ]


@router.delete("/configs/{config_id}")
async def delete_config(
    config_id: str,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """删除配置"""
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    config_dao = _auth_service.db_manager.get_dao('watermark_config')
    cfg = await config_dao.get_by_id(config_id)
    if not cfg or cfg.created_by != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Config not found")

    await config_dao.delete(config_id)
    return {"message": "Config deleted successfully"}


# ============= 审计日志 =============

@router.get("/audit/logs")
async def get_audit_logs(
    action: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """查询审计日志"""
    # TODO: 调用审计服务
    return []
