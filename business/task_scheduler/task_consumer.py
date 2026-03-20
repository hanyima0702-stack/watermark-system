"""
Worker 消费者
计算池内的 Worker 节点，通过竞争消费者模式从 RabbitMQ 拉取任务并执行水印处理

启动方式:
    python -m business.task_scheduler.task_consumer --queues watermark.video watermark.audio
"""

import asyncio
import os
import sys
import time
import uuid
import logging
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import aio_pika

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from business.task_scheduler.rabbitmq_client import RabbitMQClient, ALL_QUEUES
from business.task_scheduler.task_protocol import (
    WatermarkTaskMessage,
    FileCategory,
    WatermarkAction,
    EXCHANGE_NAME,
)

logger = logging.getLogger(__name__)


class WatermarkWorker:
    """
    水印处理 Worker

    - 竞争消费者模式：多个 Worker 实例监听同一队列，RabbitMQ 轮询分发
    - 消息确认：处理成功后 ack，失败后 nack + 重试/死信
    - 进度上报：通过 Redis 更新任务进度
    - 结果回写：处理完成后更新数据库任务状态
    """

    def __init__(
        self,
        rabbitmq_url: str,
        queues: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379/0",
    ):
        self._rabbitmq_url = rabbitmq_url
        self._queues = queues or list(ALL_QUEUES.keys())
        self._redis_url = redis_url
        self._mq_client = RabbitMQClient(url=rabbitmq_url)
        self._running = False
        self._redis = None
        self._db_manager = None
        self._minio_service = None

    async def start(self):
        """启动 Worker"""
        logger.info(f"Worker 启动中，监听队列: {self._queues}")

        # 初始化基础设施连接
        await self._init_infrastructure()

        # 连接 RabbitMQ
        await self._mq_client.connect()

        # 注册消费者
        for queue_name in self._queues:
            if queue_name in ALL_QUEUES:
                await self._mq_client.consume(
                    queue_name=queue_name,
                    callback=self._on_message,
                )

        self._running = True
        logger.info("Worker 已就绪，等待任务...")

        # 保持运行
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """停止 Worker"""
        self._running = False
        await self._mq_client.close()
        if self._redis:
            await self._redis.close()
        logger.info("Worker 已停止")

    async def _init_infrastructure(self):
        """初始化 Redis / MySQL / MinIO 连接"""
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)

        try:
            from storage.database_manager import DatabaseManager
            from shared.config import DatabaseConfig, get_settings
            settings = get_settings()
            db_config = settings.database
            self._db_manager = DatabaseManager(db_config.connection_url)
            await self._db_manager.initialize()
        except Exception as e:
            logger.warning(f"数据库初始化失败 (非致命): {e}")

        try:
            from storage.minio_service import MinIOService
            from shared.config import get_settings
            settings = get_settings()
            self._minio_service = MinIOService(settings.minio)
            await self._minio_service.initialize()
        except Exception as e:
            logger.warning(f"MinIO 初始化失败 (非致命): {e}")

    async def _on_message(self, message: aio_pika.IncomingMessage):
        """
        消息处理回调

        RabbitMQ 的竞争消费者模式保证每条消息只被一个 Worker 处理。
        核心原则：无论处理成功还是失败，都必须 ack 消息，避免无限重试循环。
        失败状态通过 Redis/DB 记录，不依赖 RabbitMQ 重投递。
        """
        task_id = "unknown"
        try:
            msg = WatermarkTaskMessage.from_json(message.body.decode("utf-8"))
            task_id = msg.task_id
            logger.info(f"[{task_id}] 收到任务: {msg.file_category}/{msg.action} - {msg.original_filename}")

            # 更新状态为 processing
            await self._update_task_status(task_id, "processing", progress=0.0)

            # 执行处理
            start_time = time.time()
            result = await self._process_task(msg)
            elapsed = time.time() - start_time

            # 更新状态为 completed
            await self._update_task_status(
                task_id, "completed",
                progress=100.0,
                result=result,
                processing_time=elapsed,
            )

            logger.info(f"[{task_id}] 任务完成，耗时 {elapsed:.2f}s")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{task_id}] 任务处理失败: {error_msg}\n{traceback.format_exc()}")

            # 记录失败状态（尽力而为，不影响 ack）
            try:
                await self._update_task_status(task_id, "failed", error_message=error_msg)
            except Exception:
                logger.error(f"[{task_id}] 更新失败状态也失败了")

        finally:
            # 无论成功还是失败，都 ack 消息，防止无限重试循环
            try:
                await message.ack()
            except Exception:
                pass

    async def _process_task(self, msg: WatermarkTaskMessage) -> dict:
        """
        根据文件类别和操作类型分发到对应的处理引擎

        这里是实际的重度计算逻辑入口。
        """
        if not self._minio_service:
            raise RuntimeError("MinIO 服务不可用")

        # 1. 从 MinIO 下载文件到临时目录
        file_bytes = await self._minio_service.download_file(
            msg.minio_bucket, msg.minio_object_key
        )
        suffix = Path(msg.original_filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            await self._update_task_status(msg.task_id, "processing", progress=10.0)

            category = FileCategory(msg.file_category)

            if msg.action == WatermarkAction.EMBED.value:
                result = await self._embed_watermark(msg, tmp_path, category)
            elif msg.action == WatermarkAction.EXTRACT.value:
                result = await self._extract_watermark(msg, tmp_path, category)
            else:
                raise ValueError(f"未知操作类型: {msg.action}")

            return result
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _embed_watermark(
        self, msg: WatermarkTaskMessage, tmp_path: str, category: FileCategory
    ) -> dict:
        """嵌入水印 — 按文件类别分发到对应引擎"""

        output_bucket = msg.output_bucket or self._minio_service.config.result_bucket
        output_key = msg.output_object_key or f"results/{msg.user_id}/{msg.task_id}{Path(msg.original_filename).suffix}"

        if category == FileCategory.VIDEO:
            return await self._embed_video(msg, tmp_path, output_bucket, output_key)
        elif category == FileCategory.AUDIO:
            return await self._embed_audio(msg, tmp_path, output_bucket, output_key)
        elif category == FileCategory.DOCUMENT:
            return await self._embed_document(msg, tmp_path, output_bucket, output_key)
        elif category == FileCategory.IMAGE:
            return await self._embed_image(msg, tmp_path, output_bucket, output_key)
        else:
            raise ValueError(f"不支持的文件类别: {category}")

    # ===================== 视频水印嵌入 =====================

    async def _embed_video(
            self, msg: WatermarkTaskMessage, tmp_path: str,
            output_bucket: str, output_key: str,
    ) -> dict:
        import asyncio
        from engines.media.video_invisible_watermark import VideoWatermarker
        from engines.media.video_visible_watermark import VideoVisibleWatermarkProcessor

        current_path = tmp_path
        tmp_files = []
        suffix = Path(msg.original_filename).suffix

        try:
            # ================= 暗水印 =================
            if msg.watermark_bits and msg.watermark_type in ("invisible", "both"):
                tmp_out = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
                tmp_files.append(tmp_out)
                vw = VideoWatermarker()

                # 【改造点】将耗时的暗水印嵌入推入线程池
                await asyncio.to_thread(
                    vw.embed_video_timerange,
                    input_video_path=current_path,
                    output_video_path=tmp_out,
                    watermark_text=msg.watermark_bits,
                    start_time=0,
                )

                current_path = tmp_out
                await self._update_task_status(msg.task_id, "processing", progress=50.0)

            # ================= 明水印 =================
            if msg.visible_text and msg.watermark_type in ("visible", "both"):
                sc = msg.saved_visible_cfg or {}
                tmp_vis = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
                tmp_files.append(tmp_vis)
                vis_proc = VideoVisibleWatermarkProcessor()
                v_color = sc.get("color", "#FF0000").lstrip("#")

                # 【改造点】将耗时的视频重编码和明水印渲染推入线程池
                # (注意: to_thread 完美支持传入 kwargs 关键字参数)
                await asyncio.to_thread(
                    vis_proc.add_text_watermark,
                    input_video=current_path,
                    output_video=tmp_vis,
                    text=msg.visible_text,
                    position="center",
                    font_size=sc.get("fontSize", 36),
                    font_color=f"0x{v_color}" if len(v_color) == 6 else "white",
                    opacity=self._normalize_opacity(sc.get("opacity")),
                    rotation=sc.get("rotation", -45),
                    tiled=sc.get("layout", "tile") == "tile",
                    tile_gap_x=sc.get("tileGapX", 120),
                    tile_gap_y=sc.get("tileGapY", 80),
                )

                current_path = tmp_vis
                await self._update_task_status(msg.task_id, "processing", progress=80.0)

            # 上传结果
            return await self._upload_result(current_path, output_bucket, output_key, msg)

        finally:
            # 清理临时文件
            for f in tmp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    # ===================== 音频水印嵌入 =====================

    async def _embed_audio(
        self, msg: WatermarkTaskMessage, tmp_path: str,
        output_bucket: str, output_key: str,
    ) -> dict:
        from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker

        processor = AudioInvisibleWatermarker()
        result = await processor.process_watermark(
            input_path=tmp_path,
            minio_service=self._minio_service,
            invisible_watermark=msg.watermark_bits,
            object_key=output_key,
            bucket_name=output_bucket,
        )
        await self._update_task_status(msg.task_id, "processing", progress=90.0)
        return result

    # ===================== 文档水印嵌入 =====================

    async def _embed_document(
            self, msg: WatermarkTaskMessage, tmp_path: str,
            output_bucket: str, output_key: str,
    ) -> dict:
        import asyncio
        from engines.document.pdf_processor import PDFProcessor
        from engines.document.base_processor import WatermarkConfig as DocWMConfig
        import shutil

        working_path = Path(tmp_path)
        converted_pdf = None
        suffix = Path(msg.original_filename).suffix.lower()

        try:
            # ================= 非 PDF 先转换 =================
            if suffix != ".pdf":
                soffice_cmd = shutil.which("soffice")
                if not soffice_cmd:
                    raise RuntimeError("soffice (LibreOffice) 未安装")
                tmp_dir = working_path.parent

                # 【改造点】使用原生异步子进程，彻底消除 subprocess 带来的阻塞
                process = await asyncio.create_subprocess_exec(
                    soffice_cmd, "--headless", "--convert-to", "pdf",
                    "--outdir", str(tmp_dir), str(working_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # 挂起等待转换完成，主线程可以去处理 RabbitMQ 心跳
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    err_msg = stderr.decode('utf-8', errors='ignore')
                    raise RuntimeError(f"文档转换 PDF 失败: {err_msg}")

                converted_pdf = tmp_dir / (working_path.stem + ".pdf")
                working_path = converted_pdf

            processor = PDFProcessor()
            vis_cfg = None
            if msg.visible_text and msg.watermark_type in ("visible", "both"):
                vis_cfg = DocWMConfig({
                    "visible_watermark": {
                        "enabled": True,
                        "layers": [{
                            "type": "text", "content": msg.visible_text,
                            "font": {"size": 36, "color": "#FF0000", "opacity": 0.4},
                            "position": {"x": "center", "y": "center", "rotation": 45, "tiled": True},
                        }],
                    }
                })

            pdf_output_key = output_key.rsplit(".", 1)[0] + ".pdf" if "." in output_key else output_key + ".pdf"

            # 这里的 process_watermark 是 async 方法，保持 await
            result = await processor.process_watermark(
                file_path=working_path,
                minio_service=self._minio_service,
                invisible_watermark=msg.watermark_bits,
                visible_watermark_config=vis_cfg,
                user_id=msg.user_id,
                object_key=pdf_output_key,
                bucket_name=output_bucket,
            )
            await self._update_task_status(msg.task_id, "processing", progress=90.0)
            return {"success": True, "minio_object_key": result.minio_object_key}

        finally:
            if converted_pdf and converted_pdf.exists():
                try:
                    os.unlink(converted_pdf)
                except OSError:
                    pass

    # ===================== 图片水印嵌入 =====================

    async def _embed_image(
        self, msg: WatermarkTaskMessage, tmp_path: str,
        output_bucket: str, output_key: str,
    ) -> dict:
        from engines.image.invisible_watermark import InvisibleWatermarkProcessor

        vis_proc = None
        vis_layers = None
        if msg.visible_text and msg.watermark_type in ("visible", "both"):
            from engines.image.visible_watermark import (
                VisibleWatermarkProcessor, WatermarkLayer, WatermarkType as VWType,
                FontConfig, PositionConfig, PositionType,
            )
            sc = msg.saved_visible_cfg or {}
            vcfg = msg.visible_config or {}
            pos_map = {
                "center": PositionType.CENTER,
                "top-left": PositionType.TOP_LEFT,
                "top-right": PositionType.TOP_RIGHT,
                "bottom-left": PositionType.BOTTOM_LEFT,
                "bottom-right": PositionType.BOTTOM_RIGHT,
            }
            vis_proc = VisibleWatermarkProcessor()
            vis_layers = [WatermarkLayer(
                type=VWType.TEXT,
                content=msg.visible_text,
                font_config=FontConfig(
                    size=vcfg.get("font_size") or sc.get("fontSize", 36),
                    color=vcfg.get("color") or sc.get("color", "#FF0000"),
                    opacity=self._normalize_opacity(vcfg.get("opacity") or sc.get("opacity", 0.4)),
                ),
                position_config=PositionConfig(
                    type=pos_map.get(vcfg.get("position", "center"), PositionType.CENTER),
                    rotation=vcfg.get("rotation") or sc.get("rotation", 45),
                    spacing_x=vcfg.get("spacing_x") or sc.get("tileGapX", 100),
                    spacing_y=vcfg.get("spacing_y") or sc.get("tileGapY", 100),
                ),
                opacity=self._normalize_opacity(vcfg.get("opacity") or sc.get("opacity", 0.4)),
                tiled=vcfg.get("tiled", True),
            )]

        processor = InvisibleWatermarkProcessor()
        result = await processor.process_watermark(
            image_path=tmp_path,
            minio_service=self._minio_service,
            invisible_watermark=msg.watermark_bits,
            visible_processor=vis_proc,
            visible_layers=vis_layers,
            object_key=output_key,
            bucket_name=output_bucket,
        )
        await self._update_task_status(msg.task_id, "processing", progress=90.0)
        return {"success": True, "minio_object_key": result.minio_object_key}

    # ===================== 水印提取 =====================

    async def _extract_watermark(
            self, msg: WatermarkTaskMessage, tmp_path: str, category: FileCategory
    ) -> dict:
        """
        提取水印
        （已改造为线程池异步非阻塞版本，防止 CPU 密集型操作阻塞 RabbitMQ 心跳）
        """
        import asyncio
        from pathlib import Path

        if category == FileCategory.IMAGE:
            from engines.image.invisible_watermark import InvisibleWatermarkProcessor
            processor = InvisibleWatermarkProcessor()

            # 【改造点】将同步的 CPU 密集运算推入后台线程池
            result = await asyncio.to_thread(
                processor.extract_watermark,
                image_path=tmp_path
            )

            return {
                "success": result.success,
                "watermark_data": result.watermark_data if result.success else None,
                "confidence": result.confidence,
            }

        elif category == FileCategory.AUDIO:
            from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker
            processor = AudioInvisibleWatermarker()

            # 【改造点】推入后台线程池
            wm_str, conf = await asyncio.to_thread(
                processor.extract_file,
                tmp_path
            )

            return {
                "success": bool(wm_str),
                "watermark_data": wm_str,
                "confidence": conf
            }

        elif category == FileCategory.VIDEO:
            from engines.media.video_invisible_watermark import VideoWatermarker
            vw = VideoWatermarker()

            # 【改造点】视频逐帧提取极其耗时，必须放入线程池
            results = await asyncio.to_thread(
                vw.extract_from_timerange,
                tmp_path,
                start_time=0
            )

            success_frames = [r for r in results if r.get("success")]
            if success_frames:
                texts = [r["data"] for r in success_frames]
                most_common = max(set(texts), key=texts.count)
                return {
                    "success": True,
                    "watermark_data": most_common,
                    "confidence": len(success_frames) / max(len(results), 1)
                }
            return {"success": False, "watermark_data": None, "confidence": 0.0}

        elif category == FileCategory.DOCUMENT:
            from engines.document.pdf_processor import PDFProcessor
            processor = PDFProcessor()

            # 【改造点】推入后台线程池
            wm_data = await asyncio.to_thread(
                processor.extract_invisible_watermark,
                Path(tmp_path)
            )

            return {
                "success": bool(wm_data),
                "watermark_data": wm_data,
                "confidence": 1.0 if wm_data else 0.0
            }

        else:
            raise ValueError(f"不支持的文件类别: {category}")

    # ===================== 工具方法 =====================

    async def _upload_result(
        self, file_path: str, bucket: str, key: str, msg: WatermarkTaskMessage
    ) -> dict:
        """上传处理结果到 MinIO"""
        with open(file_path, "rb") as f:
            data = f.read()
        ext = Path(file_path).suffix.lstrip(".") or "bin"
        content_type_map = {
            "mp4": "video/mp4", "avi": "video/avi", "mov": "video/quicktime",
            "mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac",
            "pdf": "application/pdf",
        }
        content_type = content_type_map.get(ext, f"application/octet-stream")
        await self._minio_service.upload_file(
            bucket_name=bucket, object_key=key,
            file_data=data, content_type=content_type,
            metadata={"watermark": msg.watermark_bits or "", "task_id": msg.task_id},
        )
        return {"success": True, "minio_object_key": f"{bucket}/{key}"}

    async def _update_task_status(
        self, task_id: str, status: str,
        progress: float = None, result: dict = None,
        processing_time: float = None, error_message: str = None,
    ):
        """通过 Redis 更新任务状态（供前端轮询 / WebSocket 推送）"""
        import json
        if self._redis:
            try:
                data = {"status": status, "updated_at": time.time()}
                if progress is not None:
                    data["progress"] = progress
                if result:
                    data["result"] = result
                if processing_time is not None:
                    data["processing_time"] = processing_time
                if error_message:
                    data["error_message"] = error_message
                await self._redis.set(
                    f"task:{task_id}:status",
                    json.dumps(data, ensure_ascii=False),
                    ex=86400,  # 24h TTL
                )
            except Exception as e:
                logger.warning(f"Redis 状态更新失败: {e}")

        # 同时更新数据库
        if self._db_manager and status in ("completed", "failed"):
            try:
                await self._update_task_db(task_id, status, result, processing_time, error_message)
            except Exception as e:
                logger.warning(f"数据库任务状态更新失败: {e}")

    async def _update_task_db(
        self, task_id: str, status: str,
        result: dict = None, processing_time: float = None,
        error_message: str = None,
    ):
        """更新数据库中的任务记录"""
        from datetime import datetime
        async with self._db_manager.get_session() as session:
            from sqlalchemy import update
            from storage.models.watermark_task import WatermarkTask
            values = {"status": status}
            if status == "completed":
                values["progress"] = 100.0
                values["completed_at"] = datetime.utcnow()
                if result:
                    values["quality_metrics"] = result
                if processing_time:
                    values["processing_time"] = processing_time
            elif status == "failed":
                values["error_message"] = error_message
                values["completed_at"] = datetime.utcnow()
            await session.execute(
                update(WatermarkTask).where(WatermarkTask.task_id == task_id).values(**values)
            )
            await session.commit()

    @staticmethod
    def _normalize_opacity(val) -> float:
        if val is None:
            return 0.4
        val = float(val)
        return val / 100.0 if val > 1 else val
