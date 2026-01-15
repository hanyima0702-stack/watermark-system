"""
核心接口定义
定义系统各组件间的通信协议和接口规范
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WatermarkType(Enum):
    """水印类型枚举"""
    VISIBLE = "visible"
    INVISIBLE = "invisible"
    BOTH = "both"


class MediaType(Enum):
    """媒体类型枚举"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ProcessingEngine(Enum):
    """处理引擎类型"""
    DOCUMENT_ENGINE = "document"
    IMAGE_ENGINE = "image"
    MEDIA_ENGINE = "media"
    EXTRACTION_ENGINE = "extraction"


# ============= 数据传输对象 (DTOs) =============

@dataclass
class UserInfo:
    """用户信息DTO"""
    user_id: str
    username: str
    department: str
    email: str
    roles: List[str]
    created_at: datetime


@dataclass
class FileMetadata:
    """文件元数据DTO"""
    file_id: str
    original_name: str
    file_type: str
    file_hash: str
    file_size: int
    storage_path: str
    uploaded_by: str
    uploaded_at: datetime
    metadata: Dict[str, Any]


@dataclass
class WatermarkConfig:
    """水印配置DTO"""
    config_id: str
    config_name: str
    watermark_type: WatermarkType
    visible_config: Optional[Dict[str, Any]] = None
    invisible_config: Optional[Dict[str, Any]] = None
    template_variables: Optional[Dict[str, str]] = None
    is_active: bool = True


@dataclass
class WatermarkTask:
    """水印任务DTO"""
    task_id: str
    user_id: str
    file_id: str
    task_type: str
    status: TaskStatus
    config: WatermarkConfig
    progress: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


@dataclass
class ProcessingResult:
    """处理结果DTO"""
    task_id: str
    success: bool
    output_file_path: Optional[str] = None
    processing_time: float = 0.0
    quality_metrics: Optional[Dict[str, float]] = None
    error_details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    """水印提取结果DTO"""
    result_id: str
    file_id: str
    extracted_user_id: Optional[str] = None
    extracted_timestamp: Optional[datetime] = None
    confidence_score: float = 0.0
    extraction_method: str = ""
    extraction_details: Dict[str, Any] = None
    created_at: datetime = None


@dataclass
class AuditLog:
    """审计日志DTO"""
    log_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# ============= 核心接口定义 =============

class IWatermarkProcessor(ABC):
    """水印处理器接口"""
    
    @abstractmethod
    async def embed_watermark(
        self, 
        file_path: str, 
        config: WatermarkConfig,
        user_info: UserInfo
    ) -> ProcessingResult:
        """嵌入水印"""
        pass
    
    @abstractmethod
    async def extract_watermark(self, file_path: str) -> ExtractionResult:
        """提取水印"""
        pass
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """检查是否支持指定格式"""
        pass


class ITaskScheduler(ABC):
    """任务调度器接口"""
    
    @abstractmethod
    async def submit_task(self, task: WatermarkTask) -> str:
        """提交任务"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """获取任务状态"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        pass
    
    @abstractmethod
    async def get_task_progress(self, task_id: str) -> float:
        """获取任务进度"""
        pass


class IStorageService(ABC):
    """存储服务接口"""
    
    @abstractmethod
    async def upload_file(
        self, 
        file_data: bytes, 
        filename: str,
        user_id: str
    ) -> FileMetadata:
        """上传文件"""
        pass
    
    @abstractmethod
    async def download_file(self, file_id: str) -> bytes:
        """下载文件"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """获取文件元数据"""
        pass


class IConfigManager(ABC):
    """配置管理器接口"""
    
    @abstractmethod
    async def get_config(self, config_id: str) -> Optional[WatermarkConfig]:
        """获取配置"""
        pass
    
    @abstractmethod
    async def save_config(self, config: WatermarkConfig) -> str:
        """保存配置"""
        pass
    
    @abstractmethod
    async def list_configs(self, user_id: str) -> List[WatermarkConfig]:
        """列出用户配置"""
        pass
    
    @abstractmethod
    async def delete_config(self, config_id: str) -> bool:
        """删除配置"""
        pass


class IAuthService(ABC):
    """认证服务接口"""
    
    @abstractmethod
    async def authenticate(self, token: str) -> Optional[UserInfo]:
        """验证用户身份"""
        pass
    
    @abstractmethod
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """检查用户权限"""
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str) -> Optional[UserInfo]:
        """获取用户信息"""
        pass


class IAuditService(ABC):
    """审计服务接口"""
    
    @abstractmethod
    async def log_action(self, audit_log: AuditLog) -> None:
        """记录操作日志"""
        pass
    
    @abstractmethod
    async def query_logs(
        self, 
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """查询审计日志"""
        pass


class IKeyManager(ABC):
    """密钥管理器接口"""
    
    @abstractmethod
    async def get_encryption_key(self, key_id: str) -> Optional[bytes]:
        """获取加密密钥"""
        pass
    
    @abstractmethod
    async def get_signing_certificate(self, cert_id: str) -> Optional[bytes]:
        """获取签名证书"""
        pass
    
    @abstractmethod
    async def rotate_key(self, key_id: str) -> str:
        """轮换密钥"""
        pass


# ============= 服务间通信协议 =============

@dataclass
class ServiceRequest:
    """服务请求基类"""
    request_id: str
    timestamp: datetime
    user_id: str
    service_name: str
    method: str
    parameters: Dict[str, Any]


@dataclass
class ServiceResponse:
    """服务响应基类"""
    request_id: str
    timestamp: datetime
    success: bool
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


class IMessageBroker(ABC):
    """消息代理接口"""
    
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """发布消息"""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback) -> None:
        """订阅消息"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """取消订阅"""
        pass


# ============= 事件定义 =============

@dataclass
class TaskCreatedEvent:
    """任务创建事件"""
    task_id: str
    user_id: str
    file_id: str
    config_id: str
    timestamp: datetime


@dataclass
class TaskCompletedEvent:
    """任务完成事件"""
    task_id: str
    success: bool
    output_file_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = None


@dataclass
class FileUploadedEvent:
    """文件上传事件"""
    file_id: str
    user_id: str
    file_name: str
    file_size: int
    timestamp: datetime


@dataclass
class WatermarkExtractedEvent:
    """水印提取事件"""
    file_id: str
    extracted_user_id: Optional[str]
    confidence_score: float
    extraction_method: str
    timestamp: datetime