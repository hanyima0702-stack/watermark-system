"""
MinIO对象存储服务
提供MinIO文件存储的核心功能
"""

import asyncio
import logging
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime, timedelta
from io import BytesIO

from minio import Minio
from minio.error import S3Error

# 兼容不同版本的minio库
try:
    from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists
except ImportError:
    # 如果导入失败，创建兼容的异常类
    class BucketAlreadyOwnedByYou(Exception):
        pass
    
    class BucketAlreadyExists(Exception):
        pass

from shared.config import MinIOConfig

logger = logging.getLogger(__name__)


class MinIOService:
    """MinIO存储服务类"""
    
    def __init__(self, config: MinIOConfig):
        """
        初始化MinIO服务
        
        Args:
            config: MinIO配置对象
        """
        self.config = config
        self._client: Optional[Minio] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化MinIO客户端并创建必要的buckets"""
        if self._initialized:
            logger.warning("MinIO服务已经初始化")
            return
        
        try:
            # 创建MinIO客户端
            self._client = Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure
            )
            
            # 测试连接
            await self._test_connection()
            
            # 创建所有必要的buckets
            buckets = [
                self.config.video_bucket,
                self.config.document_bucket,
                self.config.audio_bucket,
                self.config.image_bucket,
                self.config.result_bucket
            ]
            
            for bucket in buckets:
                if not await self.check_bucket_exists(bucket):
                    await self.create_bucket(bucket)
            
            self._initialized = True
            logger.info(f"MinIO服务初始化成功，连接到: {self.config.endpoint}")
            
        except Exception as e:
            logger.error(f"MinIO服务初始化失败: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """测试MinIO连接"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, list, self._client.list_buckets())
        except Exception as e:
            raise ConnectionError(f"无法连接到MinIO服务器: {e}")
    
    async def check_bucket_exists(self, bucket_name: str) -> bool:
        """
        检查bucket是否存在
        
        Args:
            bucket_name: bucket名称
            
        Returns:
            bool: bucket是否存在
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(
                None,
                self._client.bucket_exists,
                bucket_name
            )
            return exists
        except S3Error as e:
            logger.error(f"检查bucket存在性失败: {e}")
            return False
    
    async def create_bucket(self, bucket_name: str) -> bool:
        """
        创建bucket
        
        Args:
            bucket_name: bucket名称
            
        Returns:
            bool: 是否创建成功
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._client.make_bucket,
                bucket_name
            )
            logger.info(f"Bucket创建成功: {bucket_name}")
            return True
            
        except (BucketAlreadyOwnedByYou, BucketAlreadyExists):
            logger.info(f"Bucket已存在: {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"创建bucket失败: {e}")
            return False
    
    async def upload_file(
        self,
        bucket_name: str,
        object_key: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        上传文件到MinIO
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            file_data: 文件数据
            content_type: 内容类型
            metadata: 文件元数据
            
        Returns:
            Dict: 上传结果，包含etag、size等信息
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            # 准备元数据
            file_metadata = metadata or {}
            file_metadata.update({
                'upload-time': datetime.utcnow().isoformat(),
                'original-size': str(len(file_data))
            })
            
            # 创建文件流
            file_stream = BytesIO(file_data)
            
            # 上传文件
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._client.put_object,
                bucket_name,
                object_key,
                file_stream,
                len(file_data),
                content_type,
                file_metadata
            )
            
            logger.info(f"文件上传成功: {bucket_name}/{object_key}")
            
            return {
                'object_key': object_key,
                'bucket_name': bucket_name,
                'size': len(file_data),
                'etag': result.etag,
                'content_type': content_type,
                'metadata': file_metadata,
                'upload_time': datetime.utcnow()
            }
            
        except S3Error as e:
            logger.error(f"文件上传失败: {e}")
            raise
    
    async def download_file(self, bucket_name: str, object_key: str) -> bytes:
        """
        从MinIO下载文件
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            
        Returns:
            bytes: 文件数据
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._client.get_object,
                bucket_name,
                object_key
            )
            
            # 读取所有数据
            data = response.read()
            response.close()
            response.release_conn()
            
            logger.info(f"文件下载成功: {bucket_name}/{object_key}")
            return data
            
        except S3Error as e:
            logger.error(f"文件下载失败: {e}")
            raise
    
    async def delete_file(self, bucket_name: str, object_key: str) -> bool:
        """
        从MinIO删除文件
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            
        Returns:
            bool: 是否删除成功
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._client.remove_object,
                bucket_name,
                object_key
            )
            
            logger.info(f"文件删除成功: {bucket_name}/{object_key}")
            return True
            
        except S3Error as e:
            logger.error(f"文件删除失败: {e}")
            return False
    
    async def get_presigned_url(
        self,
        bucket_name: str,
        object_key: str,
        expires_in: int = 3600,
        method: str = "GET"
    ) -> str:
        """
        生成预签名URL
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            expires_in: 过期时间（秒）
            method: HTTP方法（GET或PUT）
            
        Returns:
            str: 预签名URL
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            
            if method == "GET":
                url = await loop.run_in_executor(
                    None,
                    self._client.presigned_get_object,
                    bucket_name,
                    object_key,
                    timedelta(seconds=expires_in)
                )
            elif method == "PUT":
                url = await loop.run_in_executor(
                    None,
                    self._client.presigned_put_object,
                    bucket_name,
                    object_key,
                    timedelta(seconds=expires_in)
                )
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            logger.info(f"生成预签名URL成功: {bucket_name}/{object_key}")
            return url
            
        except S3Error as e:
            logger.error(f"生成预签名URL失败: {e}")
            raise
    
    async def file_exists(self, bucket_name: str, object_key: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            
        Returns:
            bool: 文件是否存在
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._client.stat_object,
                bucket_name,
                object_key
            )
            return True
        except S3Error:
            return False
    
    async def get_file_info(self, bucket_name: str, object_key: str) -> Optional[Dict[str, Any]]:
        """
        获取文件信息
        
        Args:
            bucket_name: bucket名称
            object_key: 对象键（文件路径）
            
        Returns:
            Dict: 文件信息，包含size、content_type、etag等
        """
        if not self._client:
            raise RuntimeError("MinIO客户端未初始化")
        
        try:
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                None,
                self._client.stat_object,
                bucket_name,
                object_key
            )
            
            return {
                'object_key': object_key,
                'bucket_name': bucket_name,
                'size': stat.size,
                'content_type': stat.content_type,
                'etag': stat.etag,
                'last_modified': stat.last_modified,
                'metadata': stat.metadata or {}
            }
            
        except S3Error as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    async def close(self) -> None:
        """关闭MinIO客户端连接"""
        self._client = None
        self._initialized = False
        logger.info("MinIO服务已关闭")
