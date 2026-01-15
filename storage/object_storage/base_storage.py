"""
基础存储服务接口
定义对象存储的通用接口和抽象方法
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import hashlib
import uuid


@dataclass
class StorageObject:
    """存储对象信息"""
    object_key: str
    bucket_name: str
    size: int
    content_type: str
    etag: str
    last_modified: datetime
    metadata: Dict[str, str]
    
    @property
    def url(self) -> str:
        """获取对象URL"""
        return f"{self.bucket_name}/{self.object_key}"


@dataclass
class UploadResult:
    """上传结果"""
    object_key: str
    bucket_name: str
    size: int
    etag: str
    content_type: str
    metadata: Dict[str, str]
    upload_time: datetime


class BaseStorageService(ABC):
    """基础存储服务抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._client = None
    
    @abstractmethod
    async def initialize(self):
        """初始化存储客户端"""
        pass
    
    @abstractmethod
    async def create_bucket(self, bucket_name: str) -> bool:
        """创建存储桶"""
        pass
    
    @abstractmethod
    async def bucket_exists(self, bucket_name: str) -> bool:
        """检查存储桶是否存在"""
        pass
    
    @abstractmethod
    async def upload_file(self, bucket_name: str, object_key: str, 
                         file_data: bytes, content_type: str = None,
                         metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件"""
        pass
    
    @abstractmethod
    async def upload_stream(self, bucket_name: str, object_key: str,
                          stream: BinaryIO, size: int, content_type: str = None,
                          metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件流"""
        pass
    
    @abstractmethod
    async def download_file(self, bucket_name: str, object_key: str) -> bytes:
        """下载文件"""
        pass
    
    @abstractmethod
    async def download_stream(self, bucket_name: str, object_key: str) -> BinaryIO:
        """下载文件流"""
        pass
    
    @abstractmethod
    async def delete_file(self, bucket_name: str, object_key: str) -> bool:
        """删除文件"""
        pass
    
    @abstractmethod
    async def file_exists(self, bucket_name: str, object_key: str) -> bool:
        """检查文件是否存在"""
        pass
    
    @abstractmethod
    async def get_file_info(self, bucket_name: str, object_key: str) -> Optional[StorageObject]:
        """获取文件信息"""
        pass
    
    @abstractmethod
    async def list_files(self, bucket_name: str, prefix: str = "", 
                        limit: int = 1000) -> List[StorageObject]:
        """列出文件"""
        pass
    
    @abstractmethod
    async def copy_file(self, source_bucket: str, source_key: str,
                       dest_bucket: str, dest_key: str) -> bool:
        """复制文件"""
        pass
    
    @abstractmethod
    async def generate_presigned_url(self, bucket_name: str, object_key: str,
                                   expires_in: int = 3600, method: str = "GET") -> str:
        """生成预签名URL"""
        pass
    
    @abstractmethod
    async def get_storage_stats(self, bucket_name: str) -> Dict[str, Any]:
        """获取存储统计信息"""
        pass
    
    def generate_object_key(self, filename: str, user_id: str, 
                          prefix: str = "", use_uuid: bool = True) -> str:
        """生成对象键名"""
        # 获取文件扩展名
        file_ext = ""
        if "." in filename:
            file_ext = "." + filename.split(".")[-1].lower()
        
        # 生成唯一标识
        if use_uuid:
            unique_id = str(uuid.uuid4())
        else:
            unique_id = hashlib.md5(f"{filename}{user_id}{datetime.utcnow()}".encode()).hexdigest()[:16]
        
        # 构建对象键
        parts = []
        if prefix:
            parts.append(prefix.strip("/"))
        
        # 按日期分组
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        parts.append(date_path)
        
        # 用户目录
        parts.append(user_id)
        
        # 文件名
        parts.append(f"{unique_id}{file_ext}")
        
        return "/".join(parts)
    
    def calculate_file_hash(self, file_data: bytes, algorithm: str = "sha256") -> str:
        """计算文件哈希值"""
        if algorithm == "md5":
            return hashlib.md5(file_data).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(file_data).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(file_data).hexdigest()
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    def detect_content_type(self, filename: str) -> str:
        """检测文件内容类型"""
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        content_types = {
            # 文档类型
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            
            # 图像类型
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "webp": "image/webp",
            
            # 视频类型
            "mp4": "video/mp4",
            "avi": "video/x-msvideo",
            "mov": "video/quicktime",
            "mkv": "video/x-matroska",
            "wmv": "video/x-ms-wmv",
            "flv": "video/x-flv",
            "webm": "video/webm",
            
            # 音频类型
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "aac": "audio/aac",
            "ogg": "audio/ogg",
            "wma": "audio/x-ms-wma",
            "m4a": "audio/mp4",
            
            # 其他类型
            "txt": "text/plain",
            "json": "application/json",
            "xml": "application/xml",
            "zip": "application/zip",
            "rar": "application/x-rar-compressed",
            "7z": "application/x-7z-compressed"
        }
        
        return content_types.get(ext, "application/octet-stream")
    
    def validate_file_size(self, size: int, max_size_mb: int = 500) -> bool:
        """验证文件大小"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return size <= max_size_bytes
    
    def validate_file_type(self, filename: str, allowed_extensions: List[str] = None) -> bool:
        """验证文件类型"""
        if not allowed_extensions:
            return True
        
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        return ext in [e.lower() for e in allowed_extensions]
    
    async def close(self):
        """关闭存储客户端连接"""
        if self._client:
            # 子类实现具体的关闭逻辑
            pass