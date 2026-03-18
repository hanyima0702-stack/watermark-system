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
    """
    import uuid
    import tempfile
    import os
    from pathlib import Path

    user_id = current_user["user_id"]
    task_id = str(uuid.uuid4())[:12]

    # 1. 从 MinIO 下载原始文件到临时目录
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    db_manager = _auth_service.db_manager
    async with db_manager.get_session() as session:
        from sqlalchemy import select
        from storage.models.file_metadata import FileMetadata
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.file_id == request.file_id)
        )
        file_meta = result.scalar_one_or_none()

    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    if file_meta.uploaded_by != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    bucket_name = file_meta.get_metadata("minio_bucket")
    object_key = file_meta.get_metadata("minio_object_key")
    if not bucket_name or not object_key:
        raise HTTPException(status_code=500, detail="File storage info missing")

    minio_svc = file_service.minio_service
    if not minio_svc:
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    file_bytes = await minio_svc.download_file(bucket_name, object_key)

    # 2. 写入临时文件
    suffix = Path(file_meta.original_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    minio_object_key = None
    watermark_bits = None  # 实际嵌入的64位bit串
    try:
        import secrets
        file_type = _detect_file_type(suffix.lower())
        wm_type = request.watermark_type
        invisible_note = request.invisible_text or ""  # 用户备注
        visible_text = request.visible_text or ""

        # 从数据库加载配置模板（如果指定了 config_id 且不是 default）
        saved_visible_cfg = None
        if request.config_id and request.config_id != "default":
            try:
                config_dao = _auth_service.db_manager.get_dao('watermark_config')
                db_cfg = await config_dao.get_by_id(request.config_id)
                if db_cfg and db_cfg.created_by == user_id:
                    saved_visible_cfg = db_cfg.visible_config or {}
                    # 如果前端没传 visible_text，用配置里的 text
                    if not visible_text and saved_visible_cfg.get("text"):
                        visible_text = saved_visible_cfg["text"]
            except Exception:
                pass  # 配置加载失败不影响嵌入

        # 系统自动生成不重复的64位随机bit串作为暗水印
        need_invisible = wm_type in ("invisible", "both")
        if need_invisible:
            rand_int = secrets.randbits(64)
            watermark_bits = format(rand_int, '064b')  # "0101...1010" 64位

        # 3. 根据文件类型调用对应处理器
        if file_type == "image":
            from engines.image.invisible_watermark import InvisibleWatermarkProcessor
            vis_proc = None
            vis_layers = None
            if wm_type in ("visible", "both") and visible_text:
                from engines.image.visible_watermark import (
                    VisibleWatermarkProcessor, WatermarkLayer, WatermarkType as VWType,
                    FontConfig, PositionConfig, PositionType,
                )
                # 优先级: request.visible_config > 数据库配置 > 默认值
                vcfg = request.visible_config
                sc = saved_visible_cfg or {}

                font_size = (vcfg.font_size if vcfg else None) or sc.get("fontSize") or 36
                color = (vcfg.color if vcfg else None) or sc.get("color") or "#FF0000"
                raw_opacity = (vcfg.opacity if vcfg else None) or sc.get("opacity")
                # 数据库存的 opacity 是 0-100 的百分比，需要转成 0-1
                opacity = (raw_opacity / 100.0 if raw_opacity and raw_opacity > 1 else raw_opacity) if raw_opacity is not None else 0.4
                rotation = (vcfg.rotation if vcfg else None) or sc.get("rotation") or 45.0
                tiled = (vcfg.tiled if vcfg else None) if vcfg else (sc.get("layout", "tile") == "tile")
                spacing_x = (vcfg.spacing_x if vcfg else None) or sc.get("tileGapX") or 100
                spacing_y = (vcfg.spacing_y if vcfg else None) or sc.get("tileGapY") or 100
                pos_str = (vcfg.position if vcfg else None) or "center"

                pos_map = {
                    "center": PositionType.CENTER,
                    "top-left": PositionType.TOP_LEFT,
                    "top-right": PositionType.TOP_RIGHT,
                    "bottom-left": PositionType.BOTTOM_LEFT,
                    "bottom-right": PositionType.BOTTOM_RIGHT,
                }
                pos_type = pos_map.get(pos_str, PositionType.CENTER)

                vis_proc = VisibleWatermarkProcessor()
                vis_layers = [
                    WatermarkLayer(
                        type=VWType.TEXT,
                        content=visible_text,
                        font_config=FontConfig(size=font_size, color=color, opacity=opacity),
                        position_config=PositionConfig(
                            type=pos_type,
                            rotation=rotation,
                            spacing_x=spacing_x,
                            spacing_y=spacing_y,
                        ),
                        opacity=opacity,
                        tiled=tiled,
                    )
                ]
            processor = InvisibleWatermarkProcessor()
            result_obj = await processor.process_watermark(
                image_path=tmp_path,
                minio_service=minio_svc,
                invisible_watermark=watermark_bits,
                visible_processor=vis_proc,
                visible_layers=vis_layers,
                object_key=f"results/{user_id}/{task_id}{suffix}",
                bucket_name=minio_svc.config.result_bucket,
            )
            minio_object_key = result_obj.minio_object_key

        elif file_type == "document":
            from engines.document.pdf_processor import PDFProcessor
            from engines.document.base_processor import WatermarkConfig as DocWMConfig
            processor = PDFProcessor()
            vis_cfg = None
            if visible_text and wm_type in ("visible", "both"):
                vis_cfg = DocWMConfig({
                    "visible_watermark": {
                        "enabled": True,
                        "layers": [{"type": "text", "content": visible_text,
                                    "font": {"size": 36, "color": "#FF0000", "opacity": 0.4},
                                    "position": {"x": "center", "y": "center", "rotation": 45, "tiled": True}}]
                    }
                })
            result_obj = await processor.process_watermark(
                file_path=Path(tmp_path),
                minio_service=minio_svc,
                invisible_watermark=watermark_bits,
                visible_watermark_config=vis_cfg,
                user_id=user_id,
                object_key=f"results/{user_id}/{task_id}{suffix}",
                bucket_name=minio_svc.config.result_bucket,
            )
            minio_object_key = result_obj.minio_object_key

        elif file_type == "audio":
            from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker
            processor = AudioInvisibleWatermarker()
            result_obj = await processor.process_watermark(
                input_path=tmp_path,
                minio_service=minio_svc,
                invisible_watermark=watermark_bits,
                object_key=f"results/{user_id}/{task_id}{suffix}",
                bucket_name=minio_svc.config.result_bucket,
            )
            minio_object_key = result_obj.get("minio_object_key")

        elif file_type == "video":
            from engines.media.video_invisible_watermark import VideoWatermarkProcessor
            import tempfile as _tmpmod

            current_video = tmp_path
            tmp_video_files = []

            # 视频暗水印
            if watermark_bits:
                processor = VideoWatermarkProcessor()
                tmp_invisible = processor.invisible_watermarker.embed_file(current_video, watermark_bits)
                tmp_video_files.append(tmp_invisible)
                current_video = tmp_invisible

            # 视频明水印（FFmpeg drawtext）
            if visible_text and wm_type in ("visible", "both"):
                from engines.media.video_visible_watermark import VideoVisibleWatermarkProcessor
                sc = saved_visible_cfg or {}
                v_font_size = sc.get("fontSize") or 36
                raw_op = sc.get("opacity")
                v_opacity = (raw_op / 100.0 if raw_op and raw_op > 1 else raw_op) if raw_op is not None else 0.4
                v_color = sc.get("color", "#FF0000").lstrip("#")
                # FFmpeg 颜色格式: 0xRRGGBB
                ffmpeg_color = f"0x{v_color}" if len(v_color) == 6 else "white"

                v_tiled = sc.get("layout", "tile") == "tile"
                v_rotation = sc.get("rotation") or -45
                v_gap_x = sc.get("tileGapX") or 120
                v_gap_y = sc.get("tileGapY") or 80

                tmp_visible_file = _tmpmod.NamedTemporaryFile(suffix=suffix, delete=False).name
                tmp_video_files.append(tmp_visible_file)

                vis_proc = VideoVisibleWatermarkProcessor()
                vis_proc.add_text_watermark(
                    input_video=current_video,
                    output_video=tmp_visible_file,
                    text=visible_text,
                    position="center",
                    font_size=v_font_size,
                    font_color=ffmpeg_color,
                    opacity=v_opacity,
                    rotation=v_rotation,
                    tiled=v_tiled,
                    tile_gap_x=v_gap_x,
                    tile_gap_y=v_gap_y,
                )
                current_video = tmp_visible_file

            # 上传最终视频到 MinIO
            with open(current_video, 'rb') as f:
                video_bytes = f.read()

            ext_str = suffix.lstrip('.') or 'mp4'
            target_bucket = minio_svc.config.result_bucket
            target_key = f"results/{user_id}/{task_id}{suffix}"

            await minio_svc.upload_file(
                bucket_name=target_bucket,
                object_key=target_key,
                file_data=video_bytes,
                content_type=f"video/{ext_str}",
                metadata={"watermark": watermark_bits or ""},
            )
            minio_object_key = f"{target_bucket}/{target_key}"

            # 清理视频临时文件
            for tf in tmp_video_files:
                try:
                    os.unlink(tf)
                except OSError:
                    pass

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 保存任务记录到数据库
    try:
        task_dao = _auth_service.db_manager.get_dao('watermark_task')
        from storage.models.watermark_task import WatermarkTask as WTModel
        task_record = WTModel(
            task_id=task_id,
            user_id=user_id,
            file_id=request.file_id,
            config_id=request.config_id if request.config_id != "default" else None,
            task_type=str(wm_type),
            status="completed",
            progress=100.0,
            processing_engine=file_type,
            quality_metrics={
                "minio_object_key": minio_object_key,
                "watermark_bits": watermark_bits,
                "visible_text": visible_text or None,
                "invisible_note": invisible_note or None,
            },
            completed_at=datetime.utcnow(),
        )
        await task_dao.create(task_record)
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to save task record: {e}")

    return WatermarkTaskResponse(
        task_id=task_id,
        status="completed",
        created_at=datetime.utcnow(),
        message="Watermark embedded successfully",
        minio_object_key=minio_object_key,
        watermark_bits=watermark_bits,
    )


def _detect_file_type(ext: str) -> str:
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}:
        return "image"
    if ext in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
        return "document"
    if ext in {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a"}:
        return "audio"
    if ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}:
        return "video"
    return "unknown"


@router.get("/watermark/result")
async def download_watermark_result(
    key: str,
    mode: str = "url",
    current_user = Depends(get_current_user),
):
    """
    通过 minio_object_key 获取下载链接或直接下载。
    key 格式: {bucket}/{object_key}
    mode: "url" 返回预签名URL, "stream" 直接返回文件流
    """
    parts = key.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid key format")
    bucket, obj_key = parts

    user_id = current_user["user_id"]
    expected_prefix = f"results/{user_id}/"
    if not obj_key.startswith(expected_prefix):
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_service._initialized:
        await file_service.initialize()

    minio_svc = file_service.minio_service
    if not minio_svc:
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    if mode == "url":
        filename = obj_key.split("/")[-1]
        presigned_url = await minio_svc.get_presigned_url(
            bucket_name=bucket,
            object_key=obj_key,
            expires_in=3600,
            method="GET",
            download_filename=filename,
        )
        return {"download_url": presigned_url}

    # stream 模式：直接返回文件流
    from fastapi.responses import StreamingResponse
    import io
    file_bytes = await minio_svc.download_file(bucket, obj_key)
    filename = obj_key.split("/")[-1]
    import mimetypes
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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


@router.get("/watermark/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """列出当前用户的水印任务"""
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        return []

    user_id = current_user["user_id"]
    task_dao = _auth_service.db_manager.get_dao('watermark_task')
    tasks = await task_dao.get_user_tasks(user_id, status=status, limit=limit, offset=offset)

    results = []
    for t in tasks:
        qm = t.quality_metrics or {}
        results.append({
            "task_id": t.task_id,
            "file_id": t.file_id,
            "original_name": t.file.original_name if t.file else None,
            "file_type": t.file.file_type if t.file else None,
            "watermark_type": t.task_type,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "output_file_id": qm.get("minio_object_key"),
            "output_name": None,
        })
    return results


# ============= 水印提取 =============

@router.post("/watermark/extract", response_model=ExtractionResponse)
async def extract_watermark(
    request: ExtractionRequest,
    current_user = Depends(get_current_user),
    service_client = Depends(get_service_client)
):
    """
    提取水印信息 — 从上传的文件中提取暗水印
    """
    import uuid
    import tempfile
    import os
    from pathlib import Path

    user_id = current_user["user_id"]

    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    # 1. 查询文件元数据
    db_manager = _auth_service.db_manager
    async with db_manager.get_session() as session:
        from sqlalchemy import select
        from storage.models.file_metadata import FileMetadata
        result = await session.execute(
            select(FileMetadata).where(FileMetadata.file_id == request.file_id)
        )
        file_meta = result.scalar_one_or_none()

    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    if file_meta.uploaded_by != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    bucket_name = file_meta.get_metadata("minio_bucket")
    object_key = file_meta.get_metadata("minio_object_key")
    if not bucket_name or not object_key:
        raise HTTPException(status_code=500, detail="File storage info missing")

    minio_svc = file_service.minio_service
    if not minio_svc:
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    # 2. 下载文件到临时目录
    file_bytes = await minio_svc.download_file(bucket_name, object_key)
    suffix = Path(file_meta.original_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        file_type = _detect_file_type(suffix.lower())
        extracted_data = {}
        confidence = 0.0

        if file_type == "image":
            from engines.image.invisible_watermark import InvisibleWatermarkProcessor
            processor = InvisibleWatermarkProcessor()
            result_obj = processor.extract_watermark(image_path=tmp_path)
            if result_obj.success:
                extracted_data = {
                    "watermark_data": result_obj.watermark_data,
                    "confidence": result_obj.confidence,
                    "detected_scale": result_obj.detected_scale,
                    "grid_offset": result_obj.grid_offset,
                }
                confidence = min(result_obj.confidence / 100.0, 1.0)
            else:
                extracted_data = {"error": result_obj.error_message or "未检测到水印"}

        elif file_type == "audio":
            from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker
            processor = AudioInvisibleWatermarker()
            wm_str, conf = processor.extract_file(tmp_path)
            if wm_str:
                extracted_data = {"watermark_data": wm_str}
                confidence = conf
            else:
                extracted_data = {"error": "未检测到水印"}

        elif file_type == "document":
            from engines.document.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            result_dict = processor.extract_invisible_watermark(Path(tmp_path))
            if result_dict:
                extracted_data = {
                    "watermark_data": result_dict["watermark_data"],
                    "confidence": result_dict["confidence"],
                    "page": result_dict.get("page", 0),
                }
                confidence = min(result_dict["confidence"] / 100.0, 1.0)
            else:
                extracted_data = {"error": "未检测到水印"}

        else:
            extracted_data = {"error": f"暂不支持 {file_type} 类型的水印提取"}

        # 如果提取到了水印 bit 串，去数据库匹配嵌入记录
        wm_bits = extracted_data.get("watermark_data")
        if wm_bits and "error" not in extracted_data:
            try:
                from sqlalchemy import select
                from storage.models.watermark_task import WatermarkTask as WTModel
                async with _auth_service.db_manager.get_session() as session:
                    all_tasks = await session.execute(
                        select(WTModel).where(WTModel.status == "completed")
                    )
                    for task in all_tasks.scalars().all():
                        qm = task.quality_metrics or {}
                        if qm.get("watermark_bits") == wm_bits:
                            # 查找嵌入人用户名
                            embed_username = task.user_id
                            try:
                                from storage.models.user import User
                                user_result = await session.execute(
                                    select(User).where(User.user_id == task.user_id)
                                )
                                user_obj = user_result.scalar_one_or_none()
                                if user_obj:
                                    embed_username = user_obj.username
                            except Exception:
                                pass

                            extracted_data["embed_user"] = embed_username
                            extracted_data["embed_user_id"] = task.user_id
                            extracted_data["embed_time"] = task.created_at.isoformat() if task.created_at else None
                            extracted_data["embed_task_id"] = task.task_id
                            extracted_data["invisible_note"] = qm.get("invisible_note", "")
                            extracted_data["visible_text"] = qm.get("visible_text", "")
                            extracted_data["watermark_bits"] = wm_bits
                            break
            except Exception as e:
                import structlog
                structlog.get_logger().warning(f"Failed to lookup embed record: {e}")

        result_id = str(uuid.uuid4())[:12]

        # 保存提取记录到数据库
        try:
            from storage.models.extraction_result import ExtractionResult as ERModel
            # extracted_user_id 必须是 users 表的 user_id，不是用户名
            embed_uid = extracted_data.get("embed_user_id") or None
            # confidence 可能是 numpy float，转成 Python float
            conf_float = float(confidence)
            # extraction_details 里可能有 numpy 类型，序列化前清理
            import json
            clean_details = json.loads(json.dumps(extracted_data, default=str))
            er = ERModel(
                result_id=result_id,
                file_id=request.file_id,
                extracted_user_id=embed_uid,
                confidence_score=conf_float,
                extraction_method=file_type,
                extraction_details=clean_details,
                created_at=datetime.utcnow(),
            )
            async with _auth_service.db_manager.get_session() as session:
                session.add(er)
                await session.commit()
        except Exception as e:
            import structlog
            structlog.get_logger().warning(f"Failed to save extraction record: {e}")

        return ExtractionResponse(
            result_id=result_id,
            file_id=request.file_id,
            extracted_data=extracted_data,
            confidence_score=confidence,
            extraction_time=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ============= 提取记录 =============

@router.get("/watermark/extractions")
async def list_extractions(
    limit: int = 50,
    current_user = Depends(get_current_user),
):
    """列出当前用户的提取记录"""
    from gateway.api.routers.auth import _auth_service
    if not _auth_service:
        return []

    user_id = current_user["user_id"]
    try:
        from sqlalchemy import select, desc
        from sqlalchemy.orm import selectinload
        from storage.models.extraction_result import ExtractionResult as ERModel
        from storage.models.file_metadata import FileMetadata

        async with _auth_service.db_manager.get_session() as session:
            query = (
                select(ERModel)
                .join(FileMetadata, ERModel.file_id == FileMetadata.file_id)
                .where(FileMetadata.uploaded_by == user_id)
                .options(selectinload(ERModel.file))
                .order_by(desc(ERModel.created_at))
                .limit(limit)
            )
            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "result_id": r.result_id,
                    "file_id": r.file_id,
                    "file_name": r.file.original_name if r.file else None,
                    "file_type": r.file.file_type if r.file else r.extraction_method,
                    "confidence_score": float(r.confidence_score or 0),
                    "extraction_method": r.extraction_method,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "extraction_details": r.extraction_details,
                }
                for r in records
            ]
    except Exception as e:
        import structlog
        structlog.get_logger().warning(f"Failed to list extractions: {e}")
        return []


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
