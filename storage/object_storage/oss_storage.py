"""
阿里云OSS对象存储服务实现
提供基于阿里云OSS的文件存储功能
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime
from io import BytesIO

import oss2
from oss2.exceptions import OssError, NoSuchBucket, NoSuchKey

from .base_storage import BaseStorageService, StorageObject, UploadResult

logger = logging.getLogger(__name__)


class OSSStorageService(BaseStorageService):
    """阿里云OSS存储服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', 'oss-cn-hangzhou.aliyuncs.com')
        self.access_key_id = config.get('access_key_id')
        self.access_key_secret = config.get('access_key_secret')
        self.default_bucket = config.get('default_bucket', 'watermark-files')
        
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("OSS访问密钥不能为空")
    
    async def initialize(self):
        """初始化OSS客户端"""
        try:
            # 创建认证对象
            auth = oss2.Auth(self.access_key_id, self.access_key_secret)
            
            # 创建服务对象
            self._service = oss2.Service(auth, self.endpoint)
            
            # 测试连接
            await self._test_connection()
            
            # 创建默认存储桶客户端
            self._client = oss2.Bucket(auth, self.endpoint, self.default_bucket)
            
            # 确保默认存储桶存在
            if not await self.bucket_exists(self.default_bucket):
                await self.create_bucket(self.default_bucket)
            
            logger.info(f"OSS客户端初始化成功，连接到: {self.endpoint}")
            
        except Exception as e:
            logger.error(f"OSS客户端初始化失败: {e}")
            raise
    
    async def _test_connection(self):
        """测试OSS连接"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, list, self._service.list_buckets())
        except Exception as e:
            raise ConnectionError(f"无法连接到OSS服务器: {e}")
    
    async def create_bucket(self, bucket_name: str) -> bool:
        """创建存储桶"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._service.create_bucket,
                bucket_name
            )
            logger.info(f"OSS存储桶创建成功: {bucket_name}")
            return True
            
        except OssError as e:
            if e.code == 'BucketAlreadyExists':
                logger.info(f"OSS存储桶已存在: {bucket_name}")
                return True
            logger.error(f"创建OSS存储桶失败: {e}")
            return False
    
    async def bucket_exists(self, bucket_name: str) -> bool:
        """检查存储桶是否存在"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._service.get_bucket_info,
                bucket_name
            )
            return True
        except NoSuchBucket:
            return False
        except OssError as e:
            logger.error(f"检查OSS存储桶存在性失败: {e}")
            return False
    
    def _get_bucket_client(self, bucket_name: str) -> oss2.Bucket:
        """获取指定存储桶的客户端"""
        if bucket_name == self.default_bucket:
            return self._client
        
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        return oss2.Bucket(auth, self.endpoint, bucket_name)
    
    async def upload_file(self, bucket_name: str, object_key: str, 
                         file_data: bytes, content_type: str = None,
                         metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件"""
        try:
            if not content_type:
                content_type = self.detect_content_type(object_key)
            
            # 计算文件哈希
            file_hash = self.calculate_file_hash(file_data, "sha256")
            
            # 准备元数据
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            # 准备用户元数据
            user_metadata = metadata or {}
            user_metadata.update({
                'upload-time': datetime.utcnow().isoformat(),
                'file-hash': file_hash,
                'original-size': str(len(file_data))
            })
            
            # 添加用户元数据到headers
            for key, value in user_metadata.items():
                headers[f'x-oss-meta-{key}'] = value
            
            # 获取存储桶客户端
            bucket_client = self._get_bucket_client(bucket_name)
            
            # 上传文件
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                bucket_client.put_object,
                object_key,
                file_data,
                headers
            )
            
            logger.info(f"OSS文件上传成功: {bucket_name}/{object_key}")
            
            return UploadResult(
                object_key=object_key,
                bucket_name=bucket_name,
                size=len(file_data),
                etag=result.etag,
                content_type=content_type,
                metadata=user_metadata,
                upload_time=datetime.utcnow()
            )
            
        except OssError as e:
            logger.error(f"OSS文件上传失败: {e}")
            raise
    
    async def upload_stream(self, bucket_name: str, object_key: str,
                          stream: BinaryIO, size: int, content_type: str = None,
                          metadata: Dict[str, str] = None) -> UploadResult:
        """上传文件流"""
        try:
            if not content_type:
                content_type = self.detect_content_type(object_key)
            
            # 准备元数据
            headers = {}
            if content_type:
                headers['Content-Type'] = content_type
            
            # 准备用户元数据
            user_metadata = metadata or {}
            user_metadata.update({
                'upload-time': datetime.utcnow().isoformat(),
                'original-size': str(size)
            })
            
            # 添加用户元数据到headers
            for key, value in user_metadata.items():
                headers[f'x-oss-meta-{key}'] = value
            
            # 获取存储桶客户端
            bucket_client = self._get_bucket_client(bucket_name)
            
            # 上传文件流
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                bucket_client.put_object,
                object_key,
                stream,
                headers
            )
            
            logger.info(f"OSS文件流上传成功: {bucket_name}/{object_key}")
            
            return UploadResult(
                object_key=object_key,
                bucket_name=bucket_name,
                size=size,
                etag=result.etag,
                content_type=content_type,
                metadata=user_metadata,
                upload_time=datetime.utcnow()
            )
            
        except OssError as e:
            logger.error(f"OSS文件流上传失败: {e}")
            raise
    
    async def download_file(self, bucket_name: str, object_key: str) -> bytes:
        """下载文件"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                bucket_client.get_object,
                object_key
            )
            
            # 读取所有数据
            data = result.read()
            
            logger.info(f"OSS文件下载成功: {bucket_name}/{object_key}")
            return data
            
        except OssError as e:
            logger.error(f"OSS文件下载失败: {e}")
            raise
    
    async def download_stream(self, bucket_name: str, object_key: str) -> BinaryIO:
        """下载文件流"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                bucket_client.get_object,
                object_key
            )
            
            logger.info(f"OSS文件流下载成功: {bucket_name}/{object_key}")
            return result
            
        except OssError as e:
            logger.error(f"OSS文件流下载失败: {e}")
            raise
    
    async def delete_file(self, bucket_name: str, object_key: str) -> bool:
        """删除文件"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                bucket_client.delete_object,
                object_key
            )
            
            logger.info(f"OSS文件删除成功: {bucket_name}/{object_key}")
            return True
            
        except OssError as e:
            logger.error(f"OSS文件删除失败: {e}")
            return False
    
    async def file_exists(self, bucket_name: str, object_key: str) -> bool:
        """检查文件是否存在"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                bucket_client.head_object,
                object_key
            )
            return True
            
        except NoSuchKey:
            return False
        except OssError:
            return False
    
    async def get_file_info(self, bucket_name: str, object_key: str) -> Optional[StorageObject]:
        """获取文件信息"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                bucket_client.head_object,
                object_key
            )
            
            # 提取用户元数据
            metadata = {}
            for key, value in result.headers.items():
                if key.startswith('x-oss-meta-'):
                    metadata[key[11:]] = value  # 移除'x-oss-meta-'前缀
            
            return StorageObject(
                object_key=object_key,
                bucket_name=bucket_name,
                size=result.content_length,
                content_type=result.content_type,
                etag=result.etag,
                last_modified=result.last_modified,
                metadata=metadata
            )
            
        except OssError as e:
            logger.error(f"获取OSS文件信息失败: {e}")
            return None
    
    async def list_files(self, bucket_name: str, prefix: str = "", 
                        limit: int = 1000) -> List[StorageObject]:
        """列出文件"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: list(oss2.ObjectIterator(bucket_client, prefix=prefix, max_keys=limit))
            )
            
            # 转换为StorageObject
            storage_objects = []
            for obj in result:
                storage_objects.append(StorageObject(
                    object_key=obj.key,
                    bucket_name=bucket_name,
                    size=obj.size,
                    content_type="",  # ObjectIterator不返回content_type
                    etag=obj.etag,
                    last_modified=obj.last_modified,
                    metadata={}
                ))
            
            return storage_objects
            
        except OssError as e:
            logger.error(f"列出OSS文件失败: {e}")
            return []
    
    async def copy_file(self, source_bucket: str, source_key: str,
                       dest_bucket: str, dest_key: str) -> bool:
        """复制文件"""
        try:
            dest_bucket_client = self._get_bucket_client(dest_bucket)
            
            # 构建源文件路径
            source_path = f"/{source_bucket}/{source_key}"
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                dest_bucket_client.copy_object,
                source_path,
                dest_key
            )
            
            logger.info(f"OSS文件复制成功: {source_bucket}/{source_key} -> {dest_bucket}/{dest_key}")
            return True
            
        except OssError as e:
            logger.error(f"OSS文件复制失败: {e}")
            return False
    
    async def generate_presigned_url(self, bucket_name: str, object_key: str,
                                   expires_in: int = 3600, method: str = "GET") -> str:
        """生成预签名URL"""
        try:
            bucket_client = self._get_bucket_client(bucket_name)
            
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                bucket_client.sign_url,
                method,
                object_key,
                expires_in
            )
            
            return url
            
        except OssError as e:
            logger.error(f"生成OSS预签名URL失败: {e}")
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
            logger.error(f"获取OSS存储统计失败: {e}")
            return {}
    
    async def close(self):
        """关闭OSS客户端连接"""
        # OSS客户端不需要显式关闭
        self._client = None
        self._service = None
        logger.info("OSS客户端连接已关闭")