"""
存储管理器
统一管理不同的对象存储服务，提供文件元数据管理和重复检测功能
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, BinaryIO, Union
from datetime import datetime
from enum import Enum

from .base_storage import BaseStorageService, StorageObject, UploadResult
from .minio_storage import MinIOStorageService
from .oss_storage import OSSStorageService
from ..database_manager import DatabaseManager
from ..models.file_metadata import FileMetadata

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """存储类型枚举"""
    MINIO = "minio"
    OSS = "oss"


class StorageManager:
    """存储管理器"""
    
    def __init__(self, db_manager: DatabaseManager, storage_config: Dict[str, Any]):
        self.db_manager = db_manager
        self.storage_config = storage_config
        self.storage_services: Dict[str, BaseStorageService] = {}
        self.default_storage = storage_config.get('default_storage', 'minio')
        
    async def initialize(self):
        """初始化存储服务"""
        try:
            # 初始化配置的存储服务
            for storage_name, config in self.storage_config.get('services', {}).items():
                storage_type = config.get('type')
                
                if storage_type == StorageType.MINIO.value:
                    service = MinIOStorageService(config)
                elif storage_type == StorageType.OSS.value:
                    service = OSSStorageService(config)
                else:
                    logger.warning(f"不支持的存储类型: {storage_type}")
                    continue
                
                await service.initialize()
                self.storage_services[storage_name] = service
                logger.info(f"存储服务初始化成功: {storage_name} ({storage_type})")
            
            if not self.storage_services:
                raise RuntimeError("没有可用的存储服务")
            
            logger.info(f"存储管理器初始化成功，默认存储: {self.default_storage}")
            
        except Exception as e:
            logger.error(f"存储管理器初始化失败: {e}")
            raise
    
    def get_storage_service(self, storage_name: Optional[str] = None) -> BaseStorageService:
        """获取存储服务"""
        service_name = storage_name or self.default_storage
        
        if service_name not in self.storage_services:
            raise ValueError(f"存储服务不存在: {service_name}")
        
        return self.storage_services[service_name]
    
    async def upload_file(self, file_data: bytes, filename: str, user_id: str,
                         content_type: Optional[str] = None,
                         metadata: Optional[Dict[str, str]] = None,
                         storage_name: Optional[str] = None,
                         check_duplicate: bool = True) -> FileMetadata:
        """上传文件并创建元数据记录"""
        try:
            # 获取存储服务
            storage_service = self.get_storage_service(storage_name)
            
            # 计算文件哈希
            file_hash = storage_service.calculate_file_hash(file_data, "sha256")
            
            # 检查重复文件
            if check_duplicate:
                existing_file = await self._check_duplicate_file(file_hash, user_id)
                if existing_file:
                    logger.info(f"发现重复文件: {filename} -> {existing_file.file_id}")
                    return existing_file
            
            # 验证文件
            await self._validate_file(file_data, filename)
            
            # 生成对象键
            object_key = storage_service.generate_object_key(filename, user_id)
            
            # 获取默认存储桶
            bucket_name = storage_service.default_bucket
            
            # 准备元数据
            upload_metadata = metadata or {}
            upload_metadata.update({
                'user_id': user_id,
                'original_filename': filename,
                'upload_source': 'watermark_system'
            })
            
            # 上传文件
            upload_result = await storage_service.upload_file(
                bucket_name=bucket_name,
                object_key=object_key,
                file_data=file_data,
                content_type=content_type,
                metadata=upload_metadata
            )
            
            # 创建文件元数据记录
            file_metadata = FileMetadata(
                file_id=FileMetadata.generate_id(),
                original_name=filename,
                file_type=storage_service.detect_content_type(filename),
                file_hash=file_hash,
                file_size=len(file_data),
                storage_path=f"{bucket_name}/{object_key}",
                uploaded_by=user_id,
                metadata={
                    'storage_service': storage_name or self.default_storage,
                    'bucket_name': bucket_name,
                    'object_key': object_key,
                    'etag': upload_result.etag,
                    'content_type': upload_result.content_type,
                    'upload_metadata': upload_result.metadata
                }
            )
            
            # 保存到数据库
            file_dao = self.db_manager.file_metadata_dao
            saved_file = await file_dao.create(file_metadata)
            
            logger.info(f"文件上传成功: {filename} -> {saved_file.file_id}")
            return saved_file
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise
    
    async def upload_stream(self, stream: BinaryIO, size: int, filename: str, user_id: str,
                          content_type: Optional[str] = None,
                          metadata: Optional[Dict[str, str]] = None,
                          storage_name: Optional[str] = None) -> FileMetadata:
        """上传文件流并创建元数据记录"""
        try:
            # 获取存储服务
            storage_service = self.get_storage_service(storage_name)
            
            # 验证文件大小
            if not storage_service.validate_file_size(size):
                raise ValueError(f"文件大小超过限制: {size} bytes")
            
            # 生成对象键
            object_key = storage_service.generate_object_key(filename, user_id)
            
            # 获取默认存储桶
            bucket_name = storage_service.default_bucket
            
            # 准备元数据
            upload_metadata = metadata or {}
            upload_metadata.update({
                'user_id': user_id,
                'original_filename': filename,
                'upload_source': 'watermark_system'
            })
            
            # 上传文件流
            upload_result = await storage_service.upload_stream(
                bucket_name=bucket_name,
                object_key=object_key,
                stream=stream,
                size=size,
                content_type=content_type,
                metadata=upload_metadata
            )
            
            # 创建文件元数据记录
            file_metadata = FileMetadata(
                file_id=FileMetadata.generate_id(),
                original_name=filename,
                file_type=storage_service.detect_content_type(filename),
                file_hash="",  # 流上传时无法预先计算哈希
                file_size=size,
                storage_path=f"{bucket_name}/{object_key}",
                uploaded_by=user_id,
                metadata={
                    'storage_service': storage_name or self.default_storage,
                    'bucket_name': bucket_name,
                    'object_key': object_key,
                    'etag': upload_result.etag,
                    'content_type': upload_result.content_type,
                    'upload_metadata': upload_result.metadata
                }
            )
            
            # 保存到数据库
            file_dao = self.db_manager.file_metadata_dao
            saved_file = await file_dao.create(file_metadata)
            
            logger.info(f"文件流上传成功: {filename} -> {saved_file.file_id}")
            return saved_file
            
        except Exception as e:
            logger.error(f"文件流上传失败: {e}")
            raise
    
    async def download_file(self, file_id: str) -> bytes:
        """下载文件"""
        try:
            # 获取文件元数据
            file_dao = self.db_manager.file_metadata_dao
            file_metadata = await file_dao.get_by_id(file_id)
            
            if not file_metadata:
                raise FileNotFoundError(f"文件不存在: {file_id}")
            
            # 解析存储路径
            storage_info = self._parse_storage_path(file_metadata)
            
            # 获取存储服务
            storage_service = self.get_storage_service(storage_info['storage_service'])
            
            # 下载文件
            file_data = await storage_service.download_file(
                bucket_name=storage_info['bucket_name'],
                object_key=storage_info['object_key']
            )
            
            logger.info(f"文件下载成功: {file_id}")
            return file_data
            
        except Exception as e:
            logger.error(f"文件下载失败: {e}")
            raise
    
    async def download_stream(self, file_id: str) -> BinaryIO:
        """下载文件流"""
        try:
            # 获取文件元数据
            file_dao = self.db_manager.file_metadata_dao
            file_metadata = await file_dao.get_by_id(file_id)
            
            if not file_metadata:
                raise FileNotFoundError(f"文件不存在: {file_id}")
            
            # 解析存储路径
            storage_info = self._parse_storage_path(file_metadata)
            
            # 获取存储服务
            storage_service = self.get_storage_service(storage_info['storage_service'])
            
            # 下载文件流
            stream = await storage_service.download_stream(
                bucket_name=storage_info['bucket_name'],
                object_key=storage_info['object_key']
            )
            
            logger.info(f"文件流下载成功: {file_id}")
            return stream
            
        except Exception as e:
            logger.error(f"文件流下载失败: {e}")
            raise
    
    async def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        try:
            # 获取文件元数据
            file_dao = self.db_manager.file_metadata_dao
            file_metadata = await file_dao.get_by_id(file_id)
            
            if not file_metadata:
                logger.warning(f"文件元数据不存在: {file_id}")
                return False
            
            # 解析存储路径
            storage_info = self._parse_storage_path(file_metadata)
            
            # 获取存储服务
            storage_service = self.get_storage_service(storage_info['storage_service'])
            
            # 删除存储文件
            storage_deleted = await storage_service.delete_file(
                bucket_name=storage_info['bucket_name'],
                object_key=storage_info['object_key']
            )
            
            # 删除数据库记录
            db_deleted = await file_dao.delete(file_id)
            
            success = storage_deleted and db_deleted
            if success:
                logger.info(f"文件删除成功: {file_id}")
            else:
                logger.warning(f"文件删除部分失败: {file_id}, storage: {storage_deleted}, db: {db_deleted}")
            
            return success
            
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    async def file_exists(self, file_id: str) -> bool:
        """检查文件是否存在"""
        try:
            # 检查数据库记录
            file_dao = self.db_manager.file_metadata_dao
            file_metadata = await file_dao.get_by_id(file_id)
            
            if not file_metadata:
                return False
            
            # 检查存储文件
            storage_info = self._parse_storage_path(file_metadata)
            storage_service = self.get_storage_service(storage_info['storage_service'])
            
            return await storage_service.file_exists(
                bucket_name=storage_info['bucket_name'],
                object_key=storage_info['object_key']
            )
            
        except Exception as e:
            logger.error(f"检查文件存在性失败: {e}")
            return False
    
    async def get_file_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """获取文件元数据"""
        try:
            file_dao = self.db_manager.file_metadata_dao
            return await file_dao.get_by_id(file_id)
        except Exception as e:
            logger.error(f"获取文件元数据失败: {e}")
            return None
    
    async def generate_download_url(self, file_id: str, expires_in: int = 3600) -> str:
        """生成文件下载URL"""
        try:
            # 获取文件元数据
            file_dao = self.db_manager.file_metadata_dao
            file_metadata = await file_dao.get_by_id(file_id)
            
            if not file_metadata:
                raise FileNotFoundError(f"文件不存在: {file_id}")
            
            # 解析存储路径
            storage_info = self._parse_storage_path(file_metadata)
            
            # 获取存储服务
            storage_service = self.get_storage_service(storage_info['storage_service'])
            
            # 生成预签名URL
            url = await storage_service.generate_presigned_url(
                bucket_name=storage_info['bucket_name'],
                object_key=storage_info['object_key'],
                expires_in=expires_in,
                method="GET"
            )
            
            return url
            
        except Exception as e:
            logger.error(f"生成下载URL失败: {e}")
            raise
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            stats = {}
            
            # 获取各存储服务的统计信息
            for service_name, service in self.storage_services.items():
                service_stats = await service.get_storage_stats(service.default_bucket)
                stats[service_name] = service_stats
            
            # 获取数据库统计信息
            file_dao = self.db_manager.file_metadata_dao
            db_stats = await file_dao.get_file_statistics()
            stats['database'] = db_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}
    
    async def _check_duplicate_file(self, file_hash: str, user_id: str) -> Optional[FileMetadata]:
        """检查重复文件"""
        try:
            file_dao = self.db_manager.file_metadata_dao
            return await file_dao.get_by_hash(file_hash, user_id)
        except Exception as e:
            logger.error(f"检查重复文件失败: {e}")
            return None
    
    async def _validate_file(self, file_data: bytes, filename: str):
        """验证文件"""
        # 获取系统配置
        config_dao = self.db_manager.system_config_dao
        
        # 检查文件大小
        max_size_mb = await config_dao.get_int_config('max_file_size_mb', 500)
        if not self.get_storage_service().validate_file_size(len(file_data), max_size_mb):
            raise ValueError(f"文件大小超过限制: {len(file_data)} bytes (最大: {max_size_mb}MB)")
        
        # 检查文件类型
        allowed_formats = await config_dao.get_list_config('supported_document_formats', [])
        allowed_formats.extend(await config_dao.get_list_config('supported_image_formats', []))
        allowed_formats.extend(await config_dao.get_list_config('supported_video_formats', []))
        allowed_formats.extend(await config_dao.get_list_config('supported_audio_formats', []))
        
        if allowed_formats and not self.get_storage_service().validate_file_type(filename, allowed_formats):
            raise ValueError(f"不支持的文件类型: {filename}")
    
    def _parse_storage_path(self, file_metadata: FileMetadata) -> Dict[str, str]:
        """解析存储路径"""
        metadata = file_metadata.metadata or {}
        
        # 优先使用元数据中的信息
        if 'bucket_name' in metadata and 'object_key' in metadata:
            return {
                'storage_service': metadata.get('storage_service', self.default_storage),
                'bucket_name': metadata['bucket_name'],
                'object_key': metadata['object_key']
            }
        
        # 解析storage_path
        if '/' in file_metadata.storage_path:
            parts = file_metadata.storage_path.split('/', 1)
            return {
                'storage_service': metadata.get('storage_service', self.default_storage),
                'bucket_name': parts[0],
                'object_key': parts[1]
            }
        
        raise ValueError(f"无法解析存储路径: {file_metadata.storage_path}")
    
    async def close(self):
        """关闭所有存储服务连接"""
        for service_name, service in self.storage_services.items():
            try:
                await service.close()
                logger.info(f"存储服务连接已关闭: {service_name}")
            except Exception as e:
                logger.error(f"关闭存储服务连接失败: {service_name}, {e}")
        
        self.storage_services.clear()
        logger.info("存储管理器已关闭")