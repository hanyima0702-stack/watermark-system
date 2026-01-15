"""
对象存储服务单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from io import BytesIO

from storage.object_storage import StorageManager, MinIOStorageService
from storage.object_storage.base_storage import UploadResult, StorageObject
from storage.models.file_metadata import FileMetadata


class TestMinIOStorageService:
    """MinIO存储服务测试"""
    
    @pytest.fixture
    def minio_config(self):
        return {
            'endpoint': 'localhost:9000',
            'access_key': 'testkey',
            'secret_key': 'testsecret',
            'secure': False,
            'default_bucket': 'test-bucket'
        }
    
    @pytest.fixture
    def minio_service(self, minio_config):
        return MinIOStorageService(minio_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, minio_service):
        """测试初始化"""
        with patch.object(minio_service, '_test_connection', new_callable=AsyncMock):
            with patch.object(minio_service, 'bucket_exists', return_value=True):
                await minio_service.initialize()
                assert minio_service._client is not None
    
    @pytest.mark.asyncio
    async def test_upload_file(self, minio_service):
        """测试文件上传"""
        # 模拟MinIO客户端
        mock_client = Mock()
        mock_result = Mock()
        mock_result.etag = 'test-etag'
        mock_client.put_object.return_value = mock_result
        minio_service._client = mock_client
        
        file_data = b"test file content"
        result = await minio_service.upload_file(
            bucket_name="test-bucket",
            object_key="test/file.txt",
            file_data=file_data,
            content_type="text/plain"
        )
        
        assert isinstance(result, UploadResult)
        assert result.object_key == "test/file.txt"
        assert result.bucket_name == "test-bucket"
        assert result.size == len(file_data)
        assert result.etag == "test-etag"
    
    @pytest.mark.asyncio
    async def test_download_file(self, minio_service):
        """测试文件下载"""
        # 模拟MinIO客户端
        mock_client = Mock()
        mock_response = Mock()
        mock_response.read.return_value = b"test file content"
        mock_client.get_object.return_value = mock_response
        minio_service._client = mock_client
        
        result = await minio_service.download_file("test-bucket", "test/file.txt")
        
        assert result == b"test file content"
        mock_client.get_object.assert_called_once_with("test-bucket", "test/file.txt")
    
    @pytest.mark.asyncio
    async def test_delete_file(self, minio_service):
        """测试文件删除"""
        # 模拟MinIO客户端
        mock_client = Mock()
        minio_service._client = mock_client
        
        result = await minio_service.delete_file("test-bucket", "test/file.txt")
        
        assert result is True
        mock_client.remove_object.assert_called_once_with("test-bucket", "test/file.txt")
    
    @pytest.mark.asyncio
    async def test_file_exists(self, minio_service):
        """测试文件存在性检查"""
        # 模拟MinIO客户端
        mock_client = Mock()
        mock_client.stat_object.return_value = Mock()
        minio_service._client = mock_client
        
        result = await minio_service.file_exists("test-bucket", "test/file.txt")
        
        assert result is True
        mock_client.stat_object.assert_called_once_with("test-bucket", "test/file.txt")
    
    def test_generate_object_key(self, minio_service):
        """测试对象键生成"""
        object_key = minio_service.generate_object_key(
            filename="test.txt",
            user_id="user123",
            prefix="uploads"
        )
        
        assert object_key.startswith("uploads/")
        assert "user123" in object_key
        assert object_key.endswith(".txt")
    
    def test_calculate_file_hash(self, minio_service):
        """测试文件哈希计算"""
        file_data = b"test content"
        
        md5_hash = minio_service.calculate_file_hash(file_data, "md5")
        sha256_hash = minio_service.calculate_file_hash(file_data, "sha256")
        
        assert len(md5_hash) == 32
        assert len(sha256_hash) == 64
    
    def test_detect_content_type(self, minio_service):
        """测试内容类型检测"""
        assert minio_service.detect_content_type("test.pdf") == "application/pdf"
        assert minio_service.detect_content_type("test.jpg") == "image/jpeg"
        assert minio_service.detect_content_type("test.mp4") == "video/mp4"
        assert minio_service.detect_content_type("test.unknown") == "application/octet-stream"
    
    def test_validate_file_size(self, minio_service):
        """测试文件大小验证"""
        # 1MB文件
        assert minio_service.validate_file_size(1024 * 1024, 2) is True
        
        # 超过限制的文件
        assert minio_service.validate_file_size(3 * 1024 * 1024, 2) is False
    
    def test_validate_file_type(self, minio_service):
        """测试文件类型验证"""
        allowed_types = ['pdf', 'jpg', 'png']
        
        assert minio_service.validate_file_type("test.pdf", allowed_types) is True
        assert minio_service.validate_file_type("test.txt", allowed_types) is False
        assert minio_service.validate_file_type("test.PDF", allowed_types) is True  # 大小写不敏感


class TestStorageManager:
    """存储管理器测试"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        db_manager.file_metadata_dao = Mock()
        db_manager.system_config_dao = Mock()
        return db_manager
    
    @pytest.fixture
    def storage_config(self):
        return {
            'default_storage': 'minio',
            'services': {
                'minio': {
                    'type': 'minio',
                    'endpoint': 'localhost:9000',
                    'access_key': 'testkey',
                    'secret_key': 'testsecret',
                    'default_bucket': 'test-bucket'
                }
            }
        }
    
    @pytest.fixture
    def storage_manager(self, mock_db_manager, storage_config):
        return StorageManager(mock_db_manager, storage_config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, storage_manager):
        """测试存储管理器初始化"""
        with patch.object(MinIOStorageService, 'initialize', new_callable=AsyncMock):
            await storage_manager.initialize()
            assert 'minio' in storage_manager.storage_services
    
    @pytest.mark.asyncio
    async def test_upload_file(self, storage_manager, mock_db_manager):
        """测试文件上传"""
        # 模拟存储服务
        mock_storage = Mock()
        mock_storage.calculate_file_hash.return_value = "test-hash"
        mock_storage.generate_object_key.return_value = "test/file.txt"
        mock_storage.default_bucket = "test-bucket"
        mock_storage.detect_content_type.return_value = "text/plain"
        mock_storage.upload_file.return_value = UploadResult(
            object_key="test/file.txt",
            bucket_name="test-bucket",
            size=100,
            etag="test-etag",
            content_type="text/plain",
            metadata={},
            upload_time=datetime.utcnow()
        )
        storage_manager.storage_services['minio'] = mock_storage
        
        # 模拟数据库操作
        mock_db_manager.file_metadata_dao.get_by_hash.return_value = None
        mock_db_manager.file_metadata_dao.create.return_value = FileMetadata(
            file_id="test-file-id",
            original_name="test.txt",
            file_type="text/plain",
            file_hash="test-hash",
            file_size=100,
            storage_path="test-bucket/test/file.txt",
            uploaded_by="user123"
        )
        mock_db_manager.system_config_dao.get_int_config.return_value = 500
        mock_db_manager.system_config_dao.get_list_config.return_value = ['txt']
        
        file_data = b"test content"
        result = await storage_manager.upload_file(
            file_data=file_data,
            filename="test.txt",
            user_id="user123"
        )
        
        assert isinstance(result, FileMetadata)
        assert result.file_id == "test-file-id"
        assert result.original_name == "test.txt"
    
    @pytest.mark.asyncio
    async def test_download_file(self, storage_manager, mock_db_manager):
        """测试文件下载"""
        # 模拟文件元数据
        file_metadata = FileMetadata(
            file_id="test-file-id",
            original_name="test.txt",
            file_type="text/plain",
            file_hash="test-hash",
            file_size=100,
            storage_path="test-bucket/test/file.txt",
            uploaded_by="user123",
            metadata={
                'storage_service': 'minio',
                'bucket_name': 'test-bucket',
                'object_key': 'test/file.txt'
            }
        )
        mock_db_manager.file_metadata_dao.get_by_id.return_value = file_metadata
        
        # 模拟存储服务
        mock_storage = Mock()
        mock_storage.download_file.return_value = b"test content"
        storage_manager.storage_services['minio'] = mock_storage
        
        result = await storage_manager.download_file("test-file-id")
        
        assert result == b"test content"
        mock_storage.download_file.assert_called_once_with(
            bucket_name="test-bucket",
            object_key="test/file.txt"
        )
    
    @pytest.mark.asyncio
    async def test_delete_file(self, storage_manager, mock_db_manager):
        """测试文件删除"""
        # 模拟文件元数据
        file_metadata = FileMetadata(
            file_id="test-file-id",
            original_name="test.txt",
            file_type="text/plain",
            file_hash="test-hash",
            file_size=100,
            storage_path="test-bucket/test/file.txt",
            uploaded_by="user123",
            metadata={
                'storage_service': 'minio',
                'bucket_name': 'test-bucket',
                'object_key': 'test/file.txt'
            }
        )
        mock_db_manager.file_metadata_dao.get_by_id.return_value = file_metadata
        mock_db_manager.file_metadata_dao.delete.return_value = True
        
        # 模拟存储服务
        mock_storage = Mock()
        mock_storage.delete_file.return_value = True
        storage_manager.storage_services['minio'] = mock_storage
        
        result = await storage_manager.delete_file("test-file-id")
        
        assert result is True
        mock_storage.delete_file.assert_called_once_with(
            bucket_name="test-bucket",
            object_key="test/file.txt"
        )
        mock_db_manager.file_metadata_dao.delete.assert_called_once_with("test-file-id")
    
    @pytest.mark.asyncio
    async def test_check_duplicate_file(self, storage_manager, mock_db_manager):
        """测试重复文件检查"""
        # 模拟存在重复文件
        existing_file = FileMetadata(
            file_id="existing-file-id",
            original_name="existing.txt",
            file_type="text/plain",
            file_hash="same-hash",
            file_size=100,
            storage_path="test-bucket/existing.txt",
            uploaded_by="user123"
        )
        mock_db_manager.file_metadata_dao.get_by_hash.return_value = existing_file
        
        result = await storage_manager._check_duplicate_file("same-hash", "user123")
        
        assert result == existing_file
        mock_db_manager.file_metadata_dao.get_by_hash.assert_called_once_with("same-hash", "user123")
    
    def test_parse_storage_path(self, storage_manager):
        """测试存储路径解析"""
        # 测试使用元数据
        file_metadata = FileMetadata(
            file_id="test-id",
            original_name="test.txt",
            file_type="text/plain",
            file_hash="hash",
            file_size=100,
            storage_path="bucket/path",
            uploaded_by="user123",
            metadata={
                'storage_service': 'minio',
                'bucket_name': 'test-bucket',
                'object_key': 'test/file.txt'
            }
        )
        
        result = storage_manager._parse_storage_path(file_metadata)
        
        assert result['storage_service'] == 'minio'
        assert result['bucket_name'] == 'test-bucket'
        assert result['object_key'] == 'test/file.txt'
        
        # 测试解析storage_path
        file_metadata.metadata = {}
        file_metadata.storage_path = "bucket/path/to/file.txt"
        
        result = storage_manager._parse_storage_path(file_metadata)
        
        assert result['bucket_name'] == 'bucket'
        assert result['object_key'] == 'path/to/file.txt'


if __name__ == "__main__":
    pytest.main([__file__])