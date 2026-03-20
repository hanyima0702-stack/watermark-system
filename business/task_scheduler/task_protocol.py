"""
标准化任务协议定义
所有水印处理任务的消息格式规范，用于 API 层与 Worker 层之间的通信
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json
import uuid


class FileCategory(str, Enum):
    """文件类别 — 决定路由到哪个处理队列"""
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"


class WatermarkAction(str, Enum):
    """水印操作类型"""
    EMBED = "embed"
    EXTRACT = "extract"


class TaskPriority(int, Enum):
    """任务优先级 (值越大越优先)"""
    LOW = 0
    NORMAL = 5
    HIGH = 8
    URGENT = 10


# 大文件阈值 (字节)：超过此值走异步队列
ASYNC_THRESHOLD_BYTES = {
    FileCategory.VIDEO: 10 * 1024 * 1024,      # 10 MB
    FileCategory.AUDIO: 20 * 1024 * 1024,       # 20 MB
    FileCategory.DOCUMENT: 50 * 1024 * 1024,    # 50 MB
    FileCategory.IMAGE: 100 * 1024 * 1024,      # 100 MB (图片一般较小，阈值高)
}

# 队列名称映射
QUEUE_ROUTING = {
    FileCategory.IMAGE: "watermark.image",
    FileCategory.DOCUMENT: "watermark.document",
    FileCategory.VIDEO: "watermark.video",
    FileCategory.AUDIO: "watermark.audio",
}

# RabbitMQ Exchange
EXCHANGE_NAME = "watermark.tasks"
DEAD_LETTER_EXCHANGE = "watermark.tasks.dlx"


@dataclass
class WatermarkTaskMessage:
    """
    标准化水印任务消息协议

    这是 API Gateway 发送到 RabbitMQ 的消息体，
    Worker 节点消费后按此协议解析并执行处理。
    """
    # === 任务标识 ===
    task_id: str
    user_id: str
    file_id: str

    # === 文件信息 ===
    file_category: str                          # FileCategory value
    original_filename: str
    file_size: int                              # 字节
    minio_bucket: str
    minio_object_key: str

    # === 水印操作 ===
    action: str = WatermarkAction.EMBED.value   # embed / extract
    watermark_type: str = "both"                # visible / invisible / both

    # === 暗水印参数 ===
    watermark_bits: Optional[str] = None        # 64-bit 随机串
    invisible_note: Optional[str] = None        # 用户备注

    # === 明水印参数 ===
    visible_text: Optional[str] = None
    visible_config: Optional[Dict[str, Any]] = None

    # === 配置 ===
    config_id: Optional[str] = None
    saved_visible_cfg: Optional[Dict[str, Any]] = None

    # === 输出 ===
    output_bucket: Optional[str] = None
    output_object_key: Optional[str] = None

    # === 调度 ===
    priority: int = TaskPriority.NORMAL.value
    callback_url: Optional[str] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 3600

    # === 元数据 ===
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WatermarkTaskMessage":
        data = json.loads(raw)
        return cls(**data)

    @property
    def routing_key(self) -> str:
        """根据文件类别返回 RabbitMQ routing key"""
        return QUEUE_ROUTING.get(
            FileCategory(self.file_category),
            "watermark.default"
        )


def should_use_async(file_category: str, file_size: int) -> bool:
    """判断是否应该走异步队列处理"""
    try:
        cat = FileCategory(file_category)
    except ValueError:
        return True  # 未知类型默认走异步
    threshold = ASYNC_THRESHOLD_BYTES.get(cat, 0)
    return file_size >= threshold
