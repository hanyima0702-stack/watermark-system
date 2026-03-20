"""
File Upload/Download Service
实现大文件分片上传、断点续传、文件验证和访问控制
"""

from fastapi import UploadFile, HTTPException, status
from typing import Optional, BinaryIO, Dict, Any
import hashlib
import os
import aiofiles
import structlog
from pathlib import Path
from datetime import datetime, timedelta
import secrets
import mimetypes
import uuid

try:
    from .config import settings
except ImportError:
    from gateway.api.config import settings

try:
    from storage.minio_service import MinIOService
    from storage.database_manager import get_database
    from storage.models.file_metadata import FileMetadata
    from shared.config import get_settings
except ImportError:
    # For testing purposes
    MinIOService = None
    get_database = None
    FileMetadata = None
    get_settings = None


logger = structlog.get_logger(__name__)


class FileValidator:
    """文件验证器"""
    
    ALLOWED_EXTENSIONS = settings.ALLOWED_EXTENSIONS
    MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE
    
    # 文件类型魔数（用于验证真实文件类型）
    MAGIC_NUMBERS = {
        b'\x50\x4B\x03\x04': ['.docx', '.xlsx', '.pptx', '.zip'],
        b'\x25\x50\x44\x46': ['.pdf'],
        b'\xFF\xD8\xFF': ['.jpg', '.jpeg'],
        b'\x89\x50\x4E\x47': ['.png'],
        b'\x47\x49\x46\x38': ['.gif'],
        b'\x42\x4D': ['.bmp'],
        b'\x49\x49\x2A\x00': ['.tiff'],
        b'\x4D\x4D\x00\x2A': ['.tiff'],
        b'\x00\x00\x00\x18\x66\x74\x79\x70': ['.mp4'],
        b'\x00\x00\x00\x1C\x66\x74\x79\x70': ['.mp4'],
        b'\x52\x49\x46\x46': ['.avi', '.wav'],
        b'\x49\x44\x33': ['.mp3'],
        b'\xFF\xFB': ['.mp3'],
    }
    
    @classmethod
    def validate_extension(cls, filename: str) -> bool:
        """验证文件扩展名"""
        ext = Path(filename).suffix.lower()
        return ext in cls.ALLOWED_EXTENSIONS
    
    @classmethod
    async def validate_file_type(cls, file_data: bytes) -> Optional[str]:
        """
        通过魔数验证文件类型
        返回检测到的文件类型，如果无法识别返回None
        """
        for magic, extensions in cls.MAGIC_NUMBERS.items():
            if file_data.startswith(magic):
                return extensions[0]
        return None
    
    @classmethod
    def validate_size(cls, file_size: int) -> bool:
        """验证文件大小"""
        return file_size <= cls.MAX_FILE_SIZE
    
    @classmethod
    async def scan_file(cls, file_path: str) -> Dict[str, Any]:
        """
        安全扫描文件
        TODO: 集成病毒扫描引擎（如ClamAV）
        """
        return {
            "safe": True,
            "threats": [],
            "scan_time": datetime.utcnow().isoformat()
        }


class ChunkedUploadManager:
    """分片上传管理器"""
    
    def __init__(self, upload_dir: str = "/tmp/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.upload_dir / "chunks"
        self.chunks_dir.mkdir(exist_ok=True)
        
        # 上传会话存储（实际应该用Redis）
        self.upload_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_upload_session(
        self,
        filename: str,
        file_size: int,
        chunk_size: int,
        user_id: str
    ) -> str:
        """
        创建上传会话
        返回upload_id
        """
        upload_id = secrets.token_urlsafe(32)
        
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        
        self.upload_sessions[upload_id] = {
            "upload_id": upload_id,
            "filename": filename,
            "file_size": file_size,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "uploaded_chunks": set(),
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=24)
        }
        
        # 创建分片目录
        chunk_dir = self.chunks_dir / upload_id
        chunk_dir.mkdir(exist_ok=True)
        
        logger.info(
            "Upload session created",
            upload_id=upload_id,
            filename=filename,
            total_chunks=total_chunks
        )
        
        return upload_id
    
    async def upload_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes
    ) -> Dict[str, Any]:
        """
        上传文件分片
        支持断点续传
        """
        if upload_id not in self.upload_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Upload session not found"
            )
        
        session = self.upload_sessions[upload_id]
        
        # 检查会话是否过期
        if datetime.utcnow() > session["expires_at"]:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Upload session expired"
            )
        
        # 检查分片索引
        if chunk_index >= session["total_chunks"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid chunk index"
            )
        
        # 保存分片
        chunk_path = self.chunks_dir / upload_id / f"chunk_{chunk_index}"
        async with aiofiles.open(chunk_path, 'wb') as f:
            await f.write(chunk_data)
        
        # 记录已上传分片
        session["uploaded_chunks"].add(chunk_index)
        
        progress = len(session["uploaded_chunks"]) / session["total_chunks"]
        
        logger.info(
            "Chunk uploaded",
            upload_id=upload_id,
            chunk_index=chunk_index,
            progress=f"{progress:.2%}"
        )
        
        return {
            "upload_id": upload_id,
            "chunk_index": chunk_index,
            "uploaded_chunks": len(session["uploaded_chunks"]),
            "total_chunks": session["total_chunks"],
            "progress": progress,
            "completed": progress == 1.0
        }
    
    async def complete_upload(self, upload_id: str) -> str:
        """
        完成上传，合并所有分片
        返回最终文件路径
        """
        if upload_id not in self.upload_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Upload session not found"
            )
        
        session = self.upload_sessions[upload_id]
        
        # 检查是否所有分片都已上传
        if len(session["uploaded_chunks"]) != session["total_chunks"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Not all chunks uploaded"
            )
        
        # 合并分片
        final_path = self.upload_dir / f"{upload_id}_{session['filename']}"
        chunk_dir = self.chunks_dir / upload_id
        
        async with aiofiles.open(final_path, 'wb') as outfile:
            for i in range(session["total_chunks"]):
                chunk_path = chunk_dir / f"chunk_{i}"
                async with aiofiles.open(chunk_path, 'rb') as infile:
                    chunk_data = await infile.read()
                    await outfile.write(chunk_data)
        
        # 验证文件大小
        actual_size = os.path.getsize(final_path)
        if actual_size != session["file_size"]:
            logger.error(
                "File size mismatch",
                upload_id=upload_id,
                expected=session["file_size"],
                actual=actual_size
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File size mismatch after merge"
            )
        
        # 清理分片
        for i in range(session["total_chunks"]):
            chunk_path = chunk_dir / f"chunk_{i}"
            chunk_path.unlink(missing_ok=True)
        chunk_dir.rmdir()
        
        # 清理会话
        del self.upload_sessions[upload_id]
        
        logger.info(
            "Upload completed",
            upload_id=upload_id,
            final_path=str(final_path)
        )
        
        return str(final_path)
    
    def get_upload_status(self, upload_id: str) -> Dict[str, Any]:
        """获取上传状态"""
        if upload_id not in self.upload_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Upload session not found"
            )
        
        session = self.upload_sessions[upload_id]
        progress = len(session["uploaded_chunks"]) / session["total_chunks"]
        
        return {
            "upload_id": upload_id,
            "filename": session["filename"],
            "uploaded_chunks": len(session["uploaded_chunks"]),
            "total_chunks": session["total_chunks"],
            "progress": progress,
            "completed": progress == 1.0,
            "expires_at": session["expires_at"].isoformat()
        }
    
    async def cancel_upload(self, upload_id: str) -> bool:
        """取消上传"""
        if upload_id not in self.upload_sessions:
            return False
        
        # 清理分片文件
        chunk_dir = self.chunks_dir / upload_id
        if chunk_dir.exists():
            for chunk_file in chunk_dir.iterdir():
                chunk_file.unlink()
            chunk_dir.rmdir()
        
        # 清理会话
        del self.upload_sessions[upload_id]
        
        logger.info("Upload cancelled", upload_id=upload_id)
        return True


class FileService:
    """文件服务"""
    
    def __init__(self, minio_service: Optional[MinIOService] = None):
        self.validator = FileValidator()
        self.upload_manager = ChunkedUploadManager()
        self.download_tokens: Dict[str, Dict[str, Any]] = {}
        self.minio_service = minio_service
        self._initialized = False
    
    async def initialize(self):
        """初始化文件服务"""
        if self._initialized:
            return
        
        # 初始化MinIO服务
        if self.minio_service is None and get_settings:
            config = get_settings()
            self.minio_service = MinIOService(config.minio)
            await self.minio_service.initialize()
        
        self._initialized = True
        logger.info("文件服务初始化成功")
    
    def _get_bucket_for_file_type(self, filename: str) -> str:
        """根据文件类型获取对应的bucket"""
        ext = Path(filename).suffix.lower()
        
        # 视频文件
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
            return self.minio_service.config.video_bucket
        # 文档文件
        elif ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']:
            return self.minio_service.config.document_bucket
        # 音频文件
        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a']:
            return self.minio_service.config.audio_bucket
        # 图片文件
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
            return self.minio_service.config.image_bucket
        else:
            # 默认使用文档bucket
            return self.minio_service.config.document_bucket
    
    async def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        sha256_hash = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def upload_file(
        self,
        file: UploadFile,
        user_id: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        简单文件上传（小文件）
        集成MinIO存储和MySQL元数据保存
        """
        # 确保服务已初始化
        if not self._initialized:
            await self.initialize()
        
        # 验证文件扩展名
        if validate and not self.validator.validate_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {self.validator.ALLOWED_EXTENSIONS}"
            )
        
        # 读取文件内容
        content = await file.read()
        file_size = len(content)
        
        # 验证文件大小
        if validate and not self.validator.validate_size(file_size):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {self.validator.MAX_FILE_SIZE} bytes"
            )
        
        # 验证文件类型（魔数）
        if validate:
            detected_type = await self.validator.validate_file_type(content)
            if detected_type:
                logger.info("File type detected", filename=file.filename, type=detected_type)
        
        # 生成文件ID和对象键
        file_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        object_key = f"{user_id}/{timestamp}_{file_id}_{file.filename}"
        
        # 计算文件哈希
        file_hash = hashlib.sha256(content).hexdigest()
        
        # 安全扫描
        scan_result = await self.validator.scan_file(str(self.upload_manager.upload_dir / file.filename))
        if not scan_result["safe"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File failed security scan"
            )
        
        # 确定bucket
        bucket_name = self._get_bucket_for_file_type(file.filename)
        
        # 上传到MinIO
        minio_result = None
        db_session = None
        
        try:
            # 上传到MinIO
            content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
            from urllib.parse import quote
            minio_result = await self.minio_service.upload_file(
                bucket_name=bucket_name,
                object_key=object_key,
                file_data=content,
                content_type=content_type,
                metadata={
                    'original_filename': quote(file.filename, safe=''),
                    'user_id': user_id,
                    'file_hash': file_hash
                }
            )
            
            # 保存元数据到MySQL
            if get_database:
                db_manager = await get_database()
                async with db_manager.get_session() as session:
                    db_session = session
                    
                    # 创建文件元数据记录
                    file_metadata = FileMetadata(
                        file_id=file_id,
                        original_name=file.filename,
                        file_type=content_type or 'application/octet-stream',
                        file_hash=file_hash,
                        file_size=file_size,
                        storage_path=f"minio://{bucket_name}/{object_key}",
                        uploaded_by=user_id,
                        extra_metadata={
                            'minio_bucket': bucket_name,
                            'minio_object_key': object_key,
                            'minio_etag': minio_result.get('etag'),
                            'content_type': content_type
                        },
                        uploaded_at=datetime.utcnow()
                    )
                    
                    session.add(file_metadata)
                    await session.commit()
            
            logger.info(
                "File uploaded successfully",
                file_id=file_id,
                filename=file.filename,
                size=file_size,
                hash=file_hash,
                bucket=bucket_name,
                object_key=object_key
            )
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "file_size": file_size,
                "file_hash": file_hash,
                "storage_path": f"minio://{bucket_name}/{object_key}",
                "bucket_name": bucket_name,
                "object_key": object_key,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # 事务回滚机制
            logger.error(f"File upload failed, rolling back: {e}")
            
            # 如果MinIO上传成功但数据库保存失败，删除MinIO中的文件
            if minio_result and self.minio_service:
                try:
                    await self.minio_service.delete_file(bucket_name, object_key)
                    logger.info("Rolled back MinIO upload", object_key=object_key)
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback MinIO upload: {rollback_error}")
            
            # 重新抛出异常
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File upload failed: {str(e)}"
            )
    
    def generate_download_token(
        self,
        file_id: str,
        user_id: str,
        expires_in: int = 3600
    ) -> str:
        """
        生成下载令牌
        实现访问控制和临时链接
        """
        token = secrets.token_urlsafe(32)
        
        self.download_tokens[token] = {
            "file_id": file_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=expires_in),
            "used": False
        }
        
        logger.info(
            "Download token generated",
            token=token[:8] + "...",
            file_id=file_id,
            expires_in=expires_in
        )
        
        return token
    
    def validate_download_token(self, token: str, user_id: str) -> Optional[str]:
        """
        验证下载令牌
        返回file_id，如果无效返回None
        """
        if token not in self.download_tokens:
            return None
        
        token_data = self.download_tokens[token]
        
        # 检查是否过期
        if datetime.utcnow() > token_data["expires_at"]:
            del self.download_tokens[token]
            return None
        
        # 检查用户权限
        if token_data["user_id"] != user_id:
            return None
        
        # 检查是否已使用（一次性令牌）
        if token_data["used"]:
            return None
        
        # 标记为已使用
        token_data["used"] = True
        
        return token_data["file_id"]
    
    async def delete_file(self, file_path: str) -> bool:
        """删除文件"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info("File deleted", file_path=file_path)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete file", file_path=file_path, error=str(e))
            return False
    
    async def download_file_by_id(
        self,
        file_id: str,
        user_id: str,
        generate_presigned_url: bool = False
    ) -> Dict[str, Any]:
        """
        通过文件ID下载文件
        验证用户权限并从MinIO获取文件
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            generate_presigned_url: 是否生成预签名URL而不是直接返回文件数据
            
        Returns:
            Dict: 包含文件数据或预签名URL
        """
        # 确保服务已初始化
        if not self._initialized:
            await self.initialize()
        
        # 从数据库获取文件元数据
        if not get_database:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not available"
            )
        
        db_manager = await get_database()
        file_metadata = None
        
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(FileMetadata).where(FileMetadata.file_id == file_id)
            )
            file_metadata = result.scalar_one_or_none()
        
        if not file_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # 验证用户权限（只有上传者可以下载）
        if file_metadata.uploaded_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to download this file"
            )
        
        # 从元数据中获取MinIO信息
        bucket_name = file_metadata.get_metadata('minio_bucket')
        object_key = file_metadata.get_metadata('minio_object_key')
        
        if not bucket_name or not object_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File storage information not found"
            )
        
        try:
            if generate_presigned_url:
                # 生成预签名URL（有效期1小时）
                presigned_url = await self.minio_service.get_presigned_url(
                    bucket_name=bucket_name,
                    object_key=object_key,
                    expires_in=3600,
                    method="GET"
                )
                
                logger.info(
                    "Generated presigned URL for file download",
                    file_id=file_id,
                    user_id=user_id
                )
                
                return {
                    "file_id": file_id,
                    "filename": file_metadata.original_name,
                    "download_url": presigned_url,
                    "expires_in": 3600
                }
            else:
                # 直接从MinIO下载文件
                file_data = await self.minio_service.download_file(
                    bucket_name=bucket_name,
                    object_key=object_key
                )
                
                logger.info(
                    "File downloaded successfully",
                    file_id=file_id,
                    user_id=user_id,
                    size=len(file_data)
                )
                
                return {
                    "file_id": file_id,
                    "filename": file_metadata.original_name,
                    "content_type": file_metadata.file_type,
                    "file_data": file_data,
                    "file_size": len(file_data)
                }
                
        except Exception as e:
            logger.error(f"Failed to download file: {e}", file_id=file_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download file: {str(e)}"
            )
    
    async def delete_file_by_id(
        self,
        file_id: str,
        user_id: str
    ) -> bool:
        """
        通过文件ID删除文件
        验证用户权限，从MinIO删除文件，从MySQL删除元数据
        
        Args:
            file_id: 文件ID
            user_id: 用户ID
            
        Returns:
            bool: 是否删除成功
        """
        # 确保服务已初始化
        if not self._initialized:
            await self.initialize()
        
        # 从数据库获取文件元数据
        if not get_database:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not available"
            )
        
        db_manager = await get_database()
        file_metadata = None
        
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(FileMetadata).where(FileMetadata.file_id == file_id)
            )
            file_metadata = result.scalar_one_or_none()
            
            if not file_metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # 验证用户权限（只有上传者可以删除）
            if file_metadata.uploaded_by != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to delete this file"
                )
            
            # 从元数据中获取MinIO信息
            bucket_name = file_metadata.get_metadata('minio_bucket')
            object_key = file_metadata.get_metadata('minio_object_key')
            
            if not bucket_name or not object_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="File storage information not found"
                )
            
            try:
                # 从MinIO删除文件
                minio_deleted = await self.minio_service.delete_file(
                    bucket_name=bucket_name,
                    object_key=object_key
                )
                
                if not minio_deleted:
                    logger.warning(
                        "Failed to delete file from MinIO, but continuing with metadata deletion",
                        file_id=file_id,
                        object_key=object_key
                    )
                
                # 从MySQL删除元数据
                await session.delete(file_metadata)
                await session.commit()
                
                logger.info(
                    "File deleted successfully",
                    file_id=file_id,
                    user_id=user_id,
                    bucket=bucket_name,
                    object_key=object_key
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete file: {e}", file_id=file_id)
                await session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete file: {str(e)}"
                )


# 全局文件服务实例
file_service = FileService()


# 初始化函数
async def init_file_service(minio_service: Optional[MinIOService] = None):
    """初始化文件服务"""
    global file_service
    if minio_service:
        file_service = FileService(minio_service)
    await file_service.initialize()
    return file_service
