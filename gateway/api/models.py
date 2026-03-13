"""
API Models
API请求和响应模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class WatermarkType(str, Enum):
    """水印类型"""
    VISIBLE = "visible"
    INVISIBLE = "invisible"
    BOTH = "both"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============= 请求模型 =============

class WatermarkTaskRequest(BaseModel):
    """水印任务请求"""
    file_id: str = Field(..., description="文件ID")
    config_id: str = Field(..., description="配置ID")
    watermark_type: WatermarkType = Field(default=WatermarkType.BOTH, description="水印类型")
    priority: int = Field(default=0, description="任务优先级")
    callback_url: Optional[str] = Field(None, description="回调URL")


class ExtractionRequest(BaseModel):
    """水印提取请求"""
    file_id: str = Field(..., description="文件ID")
    extraction_methods: Optional[List[str]] = Field(None, description="提取方法列表")
    generate_report: bool = Field(default=True, description="是否生成报告")


class ConfigRequest(BaseModel):
    """配置创建请求"""
    config_name: str = Field(..., description="配置名称")
    watermark_type: WatermarkType = Field(..., description="水印类型")
    visible_config: Optional[Dict[str, Any]] = Field(None, description="明水印配置")
    invisible_config: Optional[Dict[str, Any]] = Field(None, description="暗水印配置")
    template_variables: Optional[Dict[str, Any]] = Field(None, description="模板变量")


# ============= 响应模型 =============

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str
    filename: str
    file_size: int
    upload_time: datetime
    status: str


class WatermarkTaskResponse(BaseModel):
    """水印任务响应"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0, description="任务进度 0-1")
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_file_id: Optional[str] = None


class ExtractionResponse(BaseModel):
    """水印提取响应"""
    result_id: str
    file_id: str
    extracted_data: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0, description="置信度")
    extraction_time: datetime
    report_file_id: Optional[str] = None


class ConfigResponse(BaseModel):
    """配置响应"""
    config_id: str
    config_name: str
    watermark_type: Optional[WatermarkType] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool = True
    visible_config: Optional[Dict[str, Any]] = None
    invisible_config: Optional[Dict[str, Any]] = None
    template_variables: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    message: str
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=20, description="用户名（3-20个字符）")
    password: str = Field(..., min_length=8, description="密码（至少8位）")
    email: str = Field(..., description="邮箱地址")


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class ChunkedUploadRequest(BaseModel):
    """分片上传请求"""
    filename: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件总大小")
    chunk_size: int = Field(default=1024*1024, description="分片大小（默认1MB）")


class ChunkedUploadResponse(BaseModel):
    """分片上传响应"""
    upload_id: str
    chunk_size: int
    total_chunks: int


class UploadStatusResponse(BaseModel):
    """上传状态响应"""
    upload_id: str
    filename: str
    uploaded_chunks: int
    total_chunks: int
    progress: float
    completed: bool
    expires_at: str


class DownloadTokenResponse(BaseModel):
    """下载令牌响应"""
    token: str
    file_id: str
    expires_in: int
    download_url: str


class UserResponse(BaseModel):
    """用户信息响应"""
    id: str
    username: str
    email: str
    created_at: datetime
    last_login_at: Optional[datetime] = None
    is_active: bool = True


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class SessionResponse(BaseModel):
    """会话信息响应"""
    id: str
    created_at: datetime
    expires_at: datetime
    last_activity_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
