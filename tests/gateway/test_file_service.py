"""
File Service Unit Tests
测试文件上传下载服务
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import os
from io import BytesIO

from gateway.api.file_service import (
    FileValidator,
    ChunkedUploadManager,
    FileService
)


class TestFileValidator:
    """文件验证器测试"""
    
    def test_validate_extension_allowed(self):
        """测试允许的文件扩展名"""
        assert FileValidator.validate_extension("test.pdf")
        assert FileValidator.validate_extension("test.docx")
        assert FileValidator.validate_extension("test.jpg")
        assert FileValidator.validate_extension("test.mp4")
    
    def test_validate_extension_not_allowed(self):
        """测试不允许的文件扩展名"""
        assert not FileValidator.validate_extension("test.exe")
        assert not FileValidator.validate_extension("test.sh")
        assert not FileValidator.validate_extension("test.bat")
    
    def test_validate_extension_case_insensitive(self):
        """测试扩展名大小写不敏感"""
        assert FileValidator.validate_extension("test.PDF")
        assert FileValidator.validate_extension("test.DOCX")
    
    @pytest.mark.asyncio
    async def test_validate_file_type_pdf(self):
        """测试PDF文件类型检测"""
        pdf_magic = b'\x25\x50\x44\x46'
        file_type = await FileValidator.validate_file_type(pdf_magic + b'rest of file')
        assert file_type == '.pdf'
    
    @pytest.mark.asyncio
    async def test_validate_file_type_jpeg(self):
        """测试JPEG文件类型检测"""
        jpeg_magic = b'\xFF\xD8\xFF'
        file_type = await FileValidator.validate_file_type(jpeg_magic + b'rest of file')
        assert file_type == '.jpg'
    
    @pytest.mark.asyncio
    async def test_validate_file_type_unknown(self):
        """测试未知文件类型"""
        unknown_data = b'\x00\x00\x00\x00'
        file_type = await FileValidator.validate_file_type(unknown_data)
        assert file_type is None
    
    def test_validate_size_within_limit(self):
        """测试文件大小在限制内"""
        assert FileValidator.validate_size(1024)
        assert FileValidator.validate_size(1024 * 1024)
    
    def test_validate_size_exceeds_limit(self):
        """测试文件大小超过限制"""
        assert not FileValidator.validate_size(FileValidator.MAX_FILE_SIZE + 1)


class TestChunkedUploadManager:
    """分片上传管理器测试"""
    
    @pytest.fixture
    def upload_manager(self):
        """创建临时上传管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ChunkedUploadManager(upload_dir=tmpdir)
            yield manager
    
    def test_create_upload_session(self, upload_manager):
        """测试创建上传会话"""
        upload_id = upload_manager.create_upload_session(
            filename="test.pdf",
            file_size=1024 * 1024,  # 1MB
            chunk_size=256 * 1024,  # 256KB
            user_id="user_123"
        )
        
        assert upload_id is not None
        assert len(upload_id) > 0
        assert upload_id in upload_manager.upload_sessions
        
        session = upload_manager.upload_sessions[upload_id]
        assert session["filename"] == "test.pdf"
        assert session["file_size"] == 1024 * 1024
        assert session["total_chunks"] == 4  # 1MB / 256KB = 4
    
    @pytest.mark.asyncio
    async def test_upload_chunk(self, upload_manager):
        """测试上传分片"""
        upload_id = upload_manager.create_upload_session(
            filename="test.txt",
            file_size=1024,
            chunk_size=256,
            user_id="user_123"
        )
        
        chunk_data = b"x" * 256
        result = await upload_manager.upload_chunk(
            upload_id=upload_id,
            chunk_index=0,
            chunk_data=chunk_data
        )
        
        assert result["upload_id"] == upload_id
        assert result["chunk_index"] == 0
        assert result["uploaded_chunks"] == 1
        assert result["total_chunks"] == 4
        assert result["progress"] == 0.25
        assert not result["completed"]
    
    @pytest.mark.asyncio
    async def test_upload_all_chunks(self, upload_manager):
        """测试上传所有分片"""
        file_size = 1024
        chunk_size = 256
        upload_id = upload_manager.create_upload_session(
            filename="test.txt",
            file_size=file_size,
            chunk_size=chunk_size,
            user_id="user_123"
        )
        
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            chunk_data = b"x" * chunk_size
            result = await upload_manager.upload_chunk(
                upload_id=upload_id,
                chunk_index=i,
                chunk_data=chunk_data
            )
        
        assert result["completed"]
        assert result["progress"] == 1.0
    
    @pytest.mark.asyncio
    async def test_complete_upload(self, upload_manager):
        """测试完成上传"""
        file_size = 512
        chunk_size = 256
        upload_id = upload_manager.create_upload_session(
            filename="test.txt",
            file_size=file_size,
            chunk_size=chunk_size,
            user_id="user_123"
        )
        
        # 上传所有分片
        for i in range(2):
            chunk_data = b"x" * 256
            await upload_manager.upload_chunk(
                upload_id=upload_id,
                chunk_index=i,
                chunk_data=chunk_data
            )
        
        # 完成上传
        final_path = await upload_manager.complete_upload(upload_id)
        
        assert os.path.exists(final_path)
        assert os.path.getsize(final_path) == file_size
        
        # 会话应该被清理
        assert upload_id not in upload_manager.upload_sessions
    
    def test_get_upload_status(self, upload_manager):
        """测试获取上传状态"""
        upload_id = upload_manager.create_upload_session(
            filename="test.txt",
            file_size=1024,
            chunk_size=256,
            user_id="user_123"
        )
        
        status = upload_manager.get_upload_status(upload_id)
        
        assert status["upload_id"] == upload_id
        assert status["filename"] == "test.txt"
        assert status["uploaded_chunks"] == 0
        assert status["total_chunks"] == 4
        assert status["progress"] == 0.0
        assert not status["completed"]
    
    @pytest.mark.asyncio
    async def test_cancel_upload(self, upload_manager):
        """测试取消上传"""
        upload_id = upload_manager.create_upload_session(
            filename="test.txt",
            file_size=1024,
            chunk_size=256,
            user_id="user_123"
        )
        
        # 上传一个分片
        await upload_manager.upload_chunk(
            upload_id=upload_id,
            chunk_index=0,
            chunk_data=b"x" * 256
        )
        
        # 取消上传
        success = await upload_manager.cancel_upload(upload_id)
        
        assert success
        assert upload_id not in upload_manager.upload_sessions


class TestFileService:
    """文件服务测试"""
    
    @pytest.fixture
    def file_service_instance(self):
        """创建临时文件服务"""
        with tempfile.TemporaryDirectory() as tmpdir:
            service = FileService()
            service.upload_manager = ChunkedUploadManager(upload_dir=tmpdir)
            yield service
    
    @pytest.mark.asyncio
    async def test_calculate_file_hash(self, file_service_instance):
        """测试计算文件哈希"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            file_hash = await file_service_instance.calculate_file_hash(temp_path)
            assert len(file_hash) == 64  # SHA256 hash length
            assert file_hash.isalnum()
        finally:
            os.unlink(temp_path)
    
    def test_generate_download_token(self, file_service_instance):
        """测试生成下载令牌"""
        token = file_service_instance.generate_download_token(
            file_id="file_123",
            user_id="user_123",
            expires_in=3600
        )
        
        assert token is not None
        assert len(token) > 0
        assert token in file_service_instance.download_tokens
    
    def test_validate_download_token_valid(self, file_service_instance):
        """测试验证有效的下载令牌"""
        token = file_service_instance.generate_download_token(
            file_id="file_123",
            user_id="user_123",
            expires_in=3600
        )
        
        file_id = file_service_instance.validate_download_token(token, "user_123")
        assert file_id == "file_123"
    
    def test_validate_download_token_invalid_user(self, file_service_instance):
        """测试验证令牌时用户不匹配"""
        token = file_service_instance.generate_download_token(
            file_id="file_123",
            user_id="user_123",
            expires_in=3600
        )
        
        file_id = file_service_instance.validate_download_token(token, "user_456")
        assert file_id is None
    
    def test_validate_download_token_used(self, file_service_instance):
        """测试令牌只能使用一次"""
        token = file_service_instance.generate_download_token(
            file_id="file_123",
            user_id="user_123",
            expires_in=3600
        )
        
        # 第一次使用
        file_id = file_service_instance.validate_download_token(token, "user_123")
        assert file_id == "file_123"
        
        # 第二次使用应该失败
        file_id = file_service_instance.validate_download_token(token, "user_123")
        assert file_id is None
    
    @pytest.mark.asyncio
    async def test_delete_file(self, file_service_instance):
        """测试删除文件"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        assert os.path.exists(temp_path)
        
        success = await file_service_instance.delete_file(temp_path)
        assert success
        assert not os.path.exists(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
