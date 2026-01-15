"""
MinIO对象存储服务实现
提供基于MinIO的文件存储功能
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
from io import BytesIO
import aiohttp

from minio import Minio
from minio.error import S3Error, BucketAlreadyOwnedByYou, BucketAlreadyExists
from urllib3.exceptions import MaxRetryError

from .base_storage import BaseStorageService, StorageObject, UploadResult

logger = logging.getLogger(__name__)


class MinIOStorageService(BaseStorageService):
    """MinIO存储服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', 'localhost:9000')
        self.access_key = config.get('access_key', 'minioadmin')
        self.secret_key = config.get('secret_key', 'minioadmin123')
        self.secure = config.get('secure', False)
        self.region = config.get('region', 'us-east-1')
        self.default_bucket = config.get('default_bucket', 'watermark-files')
    
    async def initialize(self):
        """初始化MinIO客户端"""
        try:
            self._client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region
            )
            
            # 测试连接
            await self._test_connection()
            
            # 确保默认存储桶存在
            if not await self.bucket_exists(self.default_bucket):
                await self.create_bucket(self.default_bucket)
            
            logger.info(f"MinIO客户端初始化成功，连接到: {self.endpoint}")
            
        except Exception as e:
            logger.error(f"MinIO客户端初始化失败: {e}")
            raise
    
    async def _test_connection(self):
        """测试MinIO连接"""
        try:
            # 在线程池中执行同步操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, list, self._client.list_buckets())
        except Exception as e:
            raise ConnectionError(f"无法连接到MinIO服务器: {e}")
    
    async def create_bucket(self, bucket_name: str) -> bool:
        """创建存储桶"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._client.make_bucket, 
                bucket_name, 
                self.region
            )
            logger.info(f"存储桶创建成功: {bucket_name}")
            return True
            
        except (BucketAlreadyOwnedByYou, BucketAlreadyExists):
            logger.info(f"存储桶已存在: {bucket_name}")
            return True
        except S3Error as e:
            logger.error(f"创建存储桶失败: {e}")
            return False
    
    async def bucket_exists(self, bucket_name: str) -> bool:
        """检查存储桶是否存在"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self._client.bucket_exists, 
                bucket_name
            )
        except S3Error as e:
            logger.error(f"检查存储桶存在性失败: {e}")
            return False
    
    async def upload_file(self, bucket_name: str, object_key: str, 
                         file_data: bytes, content_type: str = None,
                         metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件"""
        try:
            if not content_type:
                content_type = self.detect_content_type(object_key)
            
            # 计算文件哈希
            etag = self.calculate_file_hash(file_data, "md5")
            
            # 准备元数据
            file_metadata = metadata or {}
            file_metadata.update({
                'upload-time': datetime.utcnow().isoformat(),
                'file-hash': self.calculate_file_hash(file_data, "sha256"),
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
            
            return UploadResult(
                object_key=object_key,
                bucket_name=bucket_name,
                size=len(file_data),
                etag=result.etag,
                content_type=content_type,
                metadata=file_metadata,
                upload_time=datetime.utcnow()
            )
            
        except S3Error as e:
            logger.error(f"文件上传失败: {e}")
            raise
    
    async def upload_stream(self, bucket_name: str, object_key: str,
                          stream: BinaryIO, size: int, content_type: str = None,
                          metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件流"""
        try:
            if not content_type:
                content_type = self.detect_content_type(object_key)
            
            # 准备元数据
            file_metadata = metadata or {}
            file_metadata.update({
                'upload-time': datetime.utcnow().isoformat(),
                'original-size': str(size)
            })
            
            # 上传文件流
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._client.put_object,
                bucket_name,
                object_key,
                stream,
                size,
                content_type,
                file_metadata
            )
            
            logger.info(f"文件流上传成功: {bucket_name}/{object_key}")
            
            return UploadResult(
                object_key=object_key,
                bucket_name=bucket_name,
                size=size,
                etag=result.etag,
                content_type=content_type,
                metadata=file_metadata,
                upload_time=datetime.utcnow()
            )
            
        except S3Error as e:
            logger.error(f"文件流上传失败: {e}")
            raise
    
    async def download_file(self, bucket_name: str, object_key: str) -> bytes:
        """下载文件"""
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
    
    async def download_stream(self, bucket_name: str, object_key: str) -> BinaryIO:
        """下载文件流"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._client.get_object,
                bucket_name,
                object_key
            )
            
            logger.info(f"文件流下载成功: {bucket_name}/{object_key}")
            return response
            
        except S3Error as e:
            logger.error(f"文件流下载失败: {e}")
            raise
    
    async def delete_file(self, bucket_name: str, object_key: str) -> bool:
        """删除文件"""
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
    
    async def file_exists(self, bucket_name: str, object_key: str) -> bool:
        """检查文件是否存在"""
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
    
    async def get_file_info(self, bucket_name: str, object_key: str) -> Optional[StorageObject]:
        """获取文件信息"""
        try:
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                None,
                self._client.stat_object,
                bucket_name,
                object_key
            )
            
            return StorageObject(
                object_key=object_key,
                bucket_name=bucket_name,
                size=stat.size,
                content_type=stat.content_type,
                etag=stat.etag,
                last_modified=stat.last_modified,
                metadata=stat.metadata or {}
            )
            
        except S3Error as e:
            logger.error(f"获取文件信息失败: {e}")
            return None
    
    async def list_files(self, bucket_name: str, prefix: str = "", 
                        limit: int = 1000) -> List[StorageObject]:
        """列出文件"""
        try:
            loop = asyncio.get_event_loop()
            objects = await loop.run_in_executor(
                None,
                lambda: list(self._client.list_objects(
                    bucket_name, 
                    prefix=prefix, 
                    recursive=True
                ))
            )
            
            # 限制返回数量
            objects = objects[:limit]
            
            # 转换为StorageObject
            storage_objects = []
            for obj in objects:
                storage_objects.append(StorageObject(
                    object_key=obj.object_name,
                    bucket_name=bucket_name,
                    size=obj.size,
                    content_type="",  # list_objects不返回content_type
                    etag=obj.etag,
                    last_modified=obj.last_modified,
                    metadata={}
                ))
            
            return storage_objects
            
        except S3Error as e:
            logger.error(f"列出文件失败: {e}")
            return []
    
    async def copy_file(self, source_bucket: str, source_key: str,
                       dest_bucket: str, dest_key: str) -> bool:
        """复制文件"""
        try:
            from minio.commonconfig import CopySource
            
            copy_source = CopySource(source_bucket, source_key)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._client.copy_object,
                dest_bucket,
                dest_key,
                copy_source
            )
            
            logger.info(f"文件复制成功: {source_bucket}/{source_key} -> {dest_bucket}/{dest_key}")
            return True
            
        except S3Error as e:
            logger.error(f"文件复制失败: {e}")
            return False
    
    async def generate_presigned_url(self, bucket_name: str, object_key: str,
                                   expires_in: int = 3600, method: str = "GET") -> str:
        """生成预签名URL"""
        try:
            from datetime import timedelta
            
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                self._client.presigned_get_object if method == "GET" else self._client.presigned_put_object,
                bucket_name,
                object_key,
                timedelta(seconds=expires_in)
            )
            
            return url
            
        except S3Error as e:
            logger.error(f"生成预签名URL失败: {e}")
            raise
    
    async def get_storage_stats(self, bucket_name: str) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            objects = await self.list_files(bucket_name, limit=10000)
            
            total_size = sum(obj.size for obj in objects)
            total_count = len(objects)
            
            # 按文件类型统计
            type_stats = {}
            for obj in objects:
                ext = obj.object_key.split('.')[-1].lower() if '.' in obj.object_key else 'unknown'
                if ext not in type_stats:
                    type_stats[ext] = {'count': 0, 'size': 0}
                type_stats[ext]['count'] += 1
                type_stats[ext]['size'] += obj.size
            
            return {
                'bucket_name': bucket_name,
                'total_objects': total_count,
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'type_distribution': type_stats,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}
    
    async def close(self):
        """关闭MinIO客户端连接"""
        # MinIO客户端不需要显式关闭
        self._client = None
        logger.info("MinIO客户端连接已关闭")