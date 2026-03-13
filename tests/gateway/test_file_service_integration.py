"""
File Service Integration Tests
测试文件服务与MinIO和MySQL的集成
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os
from io import BytesIO
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from gateway.api.file_service import FileService
from storage.minio_service import MinIOService
from storage.models.file_metadata import FileMetadata
from shared.config import MinIOConfig


class MockUploadFile:
    """模拟FastAPI的UploadFile"""
    
    def __init__(self, filename: str, content: bytes, content_type: str = "application/octet-stream"):
        self.filename = filename
        self.content = content
        self.content_type = content_type
        self._position = 0
    
    async def read(self, size: int = -1) -> bytes:
        """读取文件内容"""
        if size == -1:
            data = self.content[self._position:]
            self._position = len(self.content)
        else:
            data = self.content[self._position:self._position + size]
            self._position += len(data)
        return data
    
    async def seek(self, position: int):
        """移动文件指针"""
        self._position = position


@pytest.fixture
def mock_minio_service():
    """创建模拟的MinIO服务"""
    config = MinIOConfig(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False,
        video_bucket="test-videos",
        document_bucket="test-documents",
        audio_bucket="test-audios",
        image_bucket="test-images",
        result_bucket="test-results"
    )
    
    service = Mock(spec=MinIOService)
    service.config = config
    service._initialized = True
    
    # 模拟上传文件
    async def mock_upload_file(bucket_name, object_key, file_data, content_type=None, metadata=None):
        return {
            'object_key': object_key,
            'bucket_name': bucket_name,
            'size': len(file_data),
            'etag': 'mock-etag-12345',
            'content_type': content_type,
            'metadata': metadata,
            'upload_time': datetime.utcnow()
        }
    
    service.upload_file = AsyncMock(side_effect=mock_upload_file)
    
    # 模拟下载文件
    async def mock_download_file(bucket_name, object_key):
        return b"mock file content"
    
    service.download_file = AsyncMock(side_effect=mock_download_file)
    
    # 模拟删除文件
    async def mock_delete_file(bucket_name, object_key):
        return True
    
    service.delete_file = AsyncMock(side_effect=mock_delete_file)
    
    # 模拟生成预签名URL
    async def mock_get_presigned_url(bucket_name, object_key, expires_in=3600, method="GET"):
        return f"http://localhost:9000/{bucket_name}/{object_key}?expires={expires_in}"
    
    service.get_presigned_url = AsyncMock(side_effect=mock_get_presigned_url)
    
    return service


@pytest.fixture
def mock_database():
    """创建模拟的数据库管理器"""
    db_manager = Mock()
    
    # 模拟会话上下文管理器
    class MockSession:
        def __init__(self):
            self.committed = False
            self.rolled_back = False
            self._file_metadata = {}
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type:
                await self.rollback()
            else:
                await self.commit()
        
        async def commit(self):
            self.committed = True
        
        async def rollback(self):
            self.rolled_back = True
        
        def add(self, obj):
            if isinstance(obj, FileMetadata):
                self._file_metadata[obj.file_id] = obj
        
        async def delete(self, obj):
            if isinstance(obj, FileMetadata) and obj.file_id in self._file_metadata:
                del self._file_metadata[obj.file_id]
        
        async def execute(self, query):
            # 模拟查询结果
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=None)
            return result
    
    mock_session = MockSession()
    
    class MockContextManager:
        async def __aenter__(self):
            return mock_session
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    db_manager.get_session = Mock(return_value=MockContextManager())
    
    return db_manager, mock_session


class TestFileServiceIntegration:
    """文件服务集成测试"""
    
    @pytest.mark.asyncio
    async def test_upload_file_with_minio_and_mysql(self, mock_minio_service, mock_database):
        """测试带认证的文件上传（集成MinIO和MySQL）"""
        db_manager, mock_session = mock_database
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 创建模拟文件
        test_content = b"This is a test PDF file content"
        mock_file = MockUploadFile(
            filename="test_document.pdf",
            content=test_content,
            content_type="application/pdf"
        )
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 上传文件
            result = await file_service.upload_file(
                file=mock_file,
                user_id="test_user_123",
                validate=True
            )
        
        # 验证返回结果
        assert result is not None
        assert "file_id" in result
        assert result["filename"] == "test_document.pdf"
        assert result["file_size"] == len(test_content)
        assert "file_hash" in result
        assert "bucket_name" in result
        assert "object_key" in result
        
        # 验证MinIO上传被调用
        mock_minio_service.upload_file.assert_called_once()
        call_args = mock_minio_service.upload_file.call_args
        assert call_args[1]["bucket_name"] == "test-documents"
        assert "test_document.pdf" in call_args[1]["object_key"]
        
        # 验证数据库会话被提交
        assert mock_session.committed
    
    @pytest.mark.asyncio
    async def test_upload_file_rollback_on_database_error(self, mock_minio_service, mock_database):
        """测试数据库错误时的事务回滚"""
        db_manager, mock_session = mock_database
        
        # 模拟数据库提交失败
        async def mock_commit_error():
            raise Exception("Database commit failed")
        
        mock_session.commit = mock_commit_error
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 创建模拟文件
        test_content = b"Test content"
        mock_file = MockUploadFile(
            filename="test.pdf",
            content=test_content,
            content_type="application/pdf"
        )
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 上传应该失败并触发回滚
            with pytest.raises(Exception):
                await file_service.upload_file(
                    file=mock_file,
                    user_id="test_user_123",
                    validate=True
                )
        
        # 验证MinIO删除被调用（回滚）
        mock_minio_service.delete_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_file_by_id(self, mock_minio_service, mock_database):
        """测试文件下载"""
        db_manager, mock_session = mock_database
        
        # 创建模拟的文件元数据
        file_metadata = FileMetadata(
            file_id="test_file_123",
            original_name="test_document.pdf",
            file_type="application/pdf",
            file_hash="abc123",
            file_size=1024,
            storage_path="minio://test-documents/user123/test.pdf",
            uploaded_by="test_user_123",
            metadata={
                'minio_bucket': 'test-documents',
                'minio_object_key': 'user123/20240101_test.pdf'
            }
        )
        
        # 模拟数据库查询返回文件元数据
        async def mock_execute(query):
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=file_metadata)
            return result
        
        mock_session.execute = mock_execute
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 下载文件
            result = await file_service.download_file_by_id(
                file_id="test_file_123",
                user_id="test_user_123",
                generate_presigned_url=False
            )
        
        # 验证返回结果
        assert result is not None
        assert result["file_id"] == "test_file_123"
        assert result["filename"] == "test_document.pdf"
        assert "file_data" in result
        assert result["file_data"] == b"mock file content"
        
        # 验证MinIO下载被调用
        mock_minio_service.download_file.assert_called_once_with(
            bucket_name='test-documents',
            object_key='user123/20240101_test.pdf'
        )
    
    @pytest.mark.asyncio
    async def test_download_file_with_presigned_url(self, mock_minio_service, mock_database):
        """测试生成预签名URL下载"""
        db_manager, mock_session = mock_database
        
        # 创建模拟的文件元数据
        file_metadata = FileMetadata(
            file_id="test_file_123",
            original_name="test_document.pdf",
            file_type="application/pdf",
            file_hash="abc123",
            file_size=1024,
            storage_path="minio://test-documents/user123/test.pdf",
            uploaded_by="test_user_123",
            metadata={
                'minio_bucket': 'test-documents',
                'minio_object_key': 'user123/20240101_test.pdf'
            }
        )
        
        # 模拟数据库查询
        async def mock_execute(query):
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=file_metadata)
            return result
        
        mock_session.execute = mock_execute
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 生成预签名URL
            result = await file_service.download_file_by_id(
                file_id="test_file_123",
                user_id="test_user_123",
                generate_presigned_url=True
            )
        
        # 验证返回结果
        assert result is not None
        assert result["file_id"] == "test_file_123"
        assert "download_url" in result
        assert "expires_in" in result
        assert result["expires_in"] == 3600
        
        # 验证MinIO生成预签名URL被调用
        mock_minio_service.get_presigned_url.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_file_permission_denied(self, mock_minio_service, mock_database):
        """测试下载文件权限验证"""
        db_manager, mock_session = mock_database
        
        # 创建模拟的文件元数据（属于其他用户）
        file_metadata = FileMetadata(
            file_id="test_file_123",
            original_name="test_document.pdf",
            file_type="application/pdf",
            file_hash="abc123",
            file_size=1024,
            storage_path="minio://test-documents/user123/test.pdf",
            uploaded_by="other_user_456",  # 不同的用户
            metadata={
                'minio_bucket': 'test-documents',
                'minio_object_key': 'user123/20240101_test.pdf'
            }
        )
        
        # 模拟数据库查询
        async def mock_execute(query):
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=file_metadata)
            return result
        
        mock_session.execute = mock_execute
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 尝试下载应该失败
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await file_service.download_file_by_id(
                    file_id="test_file_123",
                    user_id="test_user_123",  # 不同的用户
                    generate_presigned_url=False
                )
            
            assert exc_info.value.status_code == 403
    
    @pytest.mark.asyncio
    async def test_delete_file_by_id(self, mock_minio_service, mock_database):
        """测试文件删除"""
        db_manager, mock_session = mock_database
        
        # 创建模拟的文件元数据
        file_metadata = FileMetadata(
            file_id="test_file_123",
            original_name="test_document.pdf",
            file_type="application/pdf",
            file_hash="abc123",
            file_size=1024,
            storage_path="minio://test-documents/user123/test.pdf",
            uploaded_by="test_user_123",
            metadata={
                'minio_bucket': 'test-documents',
                'minio_object_key': 'user123/20240101_test.pdf'
            }
        )
        
        # 模拟数据库查询
        async def mock_execute(query):
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=file_metadata)
            return result
        
        mock_session.execute = mock_execute
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 删除文件
            success = await file_service.delete_file_by_id(
                file_id="test_file_123",
                user_id="test_user_123"
            )
        
        # 验证删除成功
        assert success is True
        
        # 验证MinIO删除被调用
        mock_minio_service.delete_file.assert_called_once_with(
            bucket_name='test-documents',
            object_key='user123/20240101_test.pdf'
        )
        
        # 验证数据库会话被提交
        assert mock_session.committed
    
    @pytest.mark.asyncio
    async def test_delete_file_permission_denied(self, mock_minio_service, mock_database):
        """测试删除文件权限验证"""
        db_manager, mock_session = mock_database
        
        # 创建模拟的文件元数据（属于其他用户）
        file_metadata = FileMetadata(
            file_id="test_file_123",
            original_name="test_document.pdf",
            file_type="application/pdf",
            file_hash="abc123",
            file_size=1024,
            storage_path="minio://test-documents/user123/test.pdf",
            uploaded_by="other_user_456",  # 不同的用户
            metadata={
                'minio_bucket': 'test-documents',
                'minio_object_key': 'user123/20240101_test.pdf'
            }
        )
        
        # 模拟数据库查询
        async def mock_execute(query):
            result = Mock()
            result.scalar_one_or_none = Mock(return_value=file_metadata)
            return result
        
        mock_session.execute = mock_execute
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 模拟get_database
        with patch('gateway.api.file_service.get_database', return_value=db_manager):
            # 尝试删除应该失败
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await file_service.delete_file_by_id(
                    file_id="test_file_123",
                    user_id="test_user_123"  # 不同的用户
                )
            
            assert exc_info.value.status_code == 403
    
    @pytest.mark.asyncio
    async def test_upload_different_file_types(self, mock_minio_service, mock_database):
        """测试上传不同类型的文件到对应的bucket"""
        db_manager, mock_session = mock_database
        
        # 创建文件服务
        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True
        
        # 测试不同文件类型
        test_cases = [
            ("test.pdf", "test-documents"),
            ("test.mp4", "test-videos"),
            ("test.mp3", "test-audios"),
            ("test.jpg", "test-images"),
        ]
        
        for filename, expected_bucket in test_cases:
            # 重置mock
            mock_minio_service.upload_file.reset_mock()
            
            # 创建模拟文件
            mock_file = MockUploadFile(
                filename=filename,
                content=b"test content",
                content_type="application/octet-stream"
            )
            
            # 模拟get_database
            with patch('gateway.api.file_service.get_database', return_value=db_manager):
                # 上传文件
                result = await file_service.upload_file(
                    file=mock_file,
                    user_id="test_user_123",
                    validate=True
                )
            
            # 验证使用了正确的bucket
            call_args = mock_minio_service.upload_file.call_args
            assert call_args[1]["bucket_name"] == expected_bucket


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
