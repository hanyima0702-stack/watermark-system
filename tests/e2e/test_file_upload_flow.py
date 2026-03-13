"""
E2E Test: 文件上传流程
需求: 2.2-2.8, 6.7, 6.8

测试完整的文件上传流程：
1. 用户登录系统
2. 上传文件
3. 验证文件保存到MinIO
4. 验证元数据保存到MySQL
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api.routers.v1 import router as v1_router
from gateway.api.routers.auth import router as auth_router, set_auth_service
from gateway.api.dependencies import get_current_user, get_service_client
from gateway.api.auth.auth_service import AuthService
from gateway.api.auth.jwt_manager import JWTManager
from gateway.api.auth.password_utils import hash_password


# ============= Helpers =============

def create_test_app(override_user=None, auth_service=None):
    """Create a test FastAPI app with optional auth override."""
    app = FastAPI()

    if override_user is not None:
        async def mock_get_current_user():
            return override_user
        app.dependency_overrides[get_current_user] = mock_get_current_user

    async def mock_get_service_client():
        return None
    app.dependency_overrides[get_service_client] = mock_get_service_client

    if auth_service:
        set_auth_service(auth_service)

    app.include_router(auth_router, tags=["Authentication"])
    app.include_router(v1_router, prefix="/api/v1")
    return app


@pytest.fixture
def authenticated_user():
    """Authenticated user data."""
    return {"user_id": "user_upload_test", "username": "uploaduser"}


@pytest.fixture
def mock_minio_service():
    """Create mock MinIO service."""
    from shared.config import MinIOConfig

    config = MinIOConfig(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin123",
        secure=False,
        video_bucket="test-videos",
        document_bucket="test-documents",
        audio_bucket="test-audios",
        image_bucket="test-images",
        result_bucket="test-results",
    )

    service = Mock()
    service.config = config
    service._initialized = True

    async def mock_upload(bucket_name, object_key, file_data, content_type=None, metadata=None):
        return {
            "object_key": object_key,
            "bucket_name": bucket_name,
            "size": len(file_data),
            "etag": "mock-etag-12345",
            "content_type": content_type,
            "metadata": metadata,
            "upload_time": datetime.utcnow(),
        }

    service.upload_file = AsyncMock(side_effect=mock_upload)

    async def mock_download(bucket_name, object_key):
        return b"mock file content"

    service.download_file = AsyncMock(side_effect=mock_download)

    async def mock_delete(bucket_name, object_key):
        return True

    service.delete_file = AsyncMock(side_effect=mock_delete)

    async def mock_presigned(bucket_name, object_key, expires_in=3600, method="GET"):
        return f"http://localhost:9000/{bucket_name}/{object_key}?expires={expires_in}"

    service.get_presigned_url = AsyncMock(side_effect=mock_presigned)

    return service


@pytest.fixture
def mock_db_manager():
    """Create mock database manager with session."""
    db_manager = Mock()

    class MockSession:
        def __init__(self):
            self.committed = False
            self.rolled_back = False
            self._records = {}

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
            self._records[getattr(obj, "file_id", id(obj))] = obj

        async def delete(self, obj):
            key = getattr(obj, "file_id", id(obj))
            self._records.pop(key, None)

        async def execute(self, query):
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


# ============= 登录 + 文件上传完整流程测试 =============

class TestFileUploadE2E:
    """测试完整的文件上传流程"""

    def test_authenticated_user_can_upload_file(
        self, authenticated_user, mock_minio_service, mock_db_manager
    ):
        """测试已登录用户可以上传文件 (需求 2.2, 6.7, 6.8)"""
        db_manager, mock_session = mock_db_manager

        app = create_test_app(override_user=authenticated_user)
        client = TestClient(app)

        # Patch file_service with our mock MinIO and DB
        with patch("gateway.api.routers.v1.file_service") as mock_file_svc:
            mock_file_svc.upload_file = AsyncMock(
                return_value={
                    "file_id": "file-uuid-001",
                    "filename": "test_document.pdf",
                    "file_size": 1024,
                    "file_hash": "abc123hash",
                    "storage_path": "minio://test-documents/user_upload_test/test.pdf",
                    "bucket_name": "test-documents",
                    "object_key": "user_upload_test/20240101_test.pdf",
                    "uploaded_at": datetime.utcnow().isoformat(),
                }
            )

            # Upload a file
            response = client.post(
                "/api/v1/files/upload",
                files={"file": ("test_document.pdf", b"PDF content here", "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == "file-uuid-001"
        assert data["filename"] == "test_document.pdf"
        assert data["status"] == "uploaded"

    def test_unauthenticated_user_cannot_upload_file(self):
        """测试未登录用户无法上传文件 (需求 6.7, 6.8)"""
        app = FastAPI()

        async def mock_get_service_client():
            return None

        app.dependency_overrides[get_service_client] = mock_get_service_client
        app.include_router(v1_router, prefix="/api/v1")

        client = TestClient(app)

        response = client.post(
            "/api/v1/files/upload",
            files={"file": ("test.pdf", b"content", "application/pdf")},
        )

        assert response.status_code == 401

    def test_upload_stores_file_in_minio(
        self, authenticated_user, mock_minio_service, mock_db_manager
    ):
        """测试文件上传后保存到MinIO (需求 2.2, 2.3, 2.4)"""
        db_manager, mock_session = mock_db_manager

        from gateway.api.file_service import FileService

        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True

        app = create_test_app(override_user=authenticated_user)
        client = TestClient(app)

        with patch("gateway.api.routers.v1.file_service", file_service), \
             patch("gateway.api.file_service.get_database", new=AsyncMock(return_value=db_manager)):

            response = client.post(
                "/api/v1/files/upload",
                files={"file": ("report.pdf", b"%PDF-1.4 test content", "application/pdf")},
            )

        assert response.status_code == 200

        # Verify MinIO upload was called
        mock_minio_service.upload_file.assert_called_once()
        call_kwargs = mock_minio_service.upload_file.call_args[1]
        assert call_kwargs["bucket_name"] == "test-documents"
        assert "user_upload_test" in call_kwargs["object_key"]
        assert call_kwargs["content_type"] == "application/pdf"

    def test_upload_stores_metadata_in_mysql(
        self, authenticated_user, mock_minio_service, mock_db_manager
    ):
        """测试文件上传后元数据保存到MySQL (需求 2.5)"""
        db_manager, mock_session = mock_db_manager

        from gateway.api.file_service import FileService

        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True

        app = create_test_app(override_user=authenticated_user)
        client = TestClient(app)

        with patch("gateway.api.routers.v1.file_service", file_service), \
             patch("gateway.api.file_service.get_database", new=AsyncMock(return_value=db_manager)):

            response = client.post(
                "/api/v1/files/upload",
                files={"file": ("data.pdf", b"%PDF-1.4 data", "application/pdf")},
            )

        assert response.status_code == 200

        # Verify metadata was saved to the database session
        assert mock_session.committed
        assert len(mock_session._records) > 0

        # Check the saved metadata
        saved_record = list(mock_session._records.values())[0]
        assert saved_record.original_name == "data.pdf"
        assert saved_record.uploaded_by == "user_upload_test"
        assert saved_record.file_size == len(b"%PDF-1.4 data")

    def test_upload_different_file_types_to_correct_buckets(
        self, authenticated_user, mock_minio_service, mock_db_manager
    ):
        """测试不同文件类型上传到对应的bucket (需求 2.2, 2.3, 2.4)"""
        db_manager, mock_session = mock_db_manager

        from gateway.api.file_service import FileService

        test_cases = [
            ("video.mp4", "test-videos"),
            ("document.pdf", "test-documents"),
            ("audio.mp3", "test-audios"),
            ("image.jpg", "test-images"),
        ]

        for filename, expected_bucket in test_cases:
            mock_minio_service.upload_file.reset_mock()
            mock_session.committed = False
            mock_session._records = {}

            file_service = FileService(minio_service=mock_minio_service)
            file_service._initialized = True

            app = create_test_app(override_user=authenticated_user)
            client = TestClient(app)

            with patch("gateway.api.routers.v1.file_service", file_service), \
                 patch("gateway.api.file_service.get_database", new=AsyncMock(return_value=db_manager)):

                response = client.post(
                    "/api/v1/files/upload",
                    files={"file": (filename, b"test content", "application/octet-stream")},
                )

            assert response.status_code == 200, f"Failed for {filename}"
            call_kwargs = mock_minio_service.upload_file.call_args[1]
            assert call_kwargs["bucket_name"] == expected_bucket, \
                f"Expected {expected_bucket} for {filename}, got {call_kwargs['bucket_name']}"

    def test_upload_rollback_on_db_failure(
        self, authenticated_user, mock_minio_service, mock_db_manager
    ):
        """测试数据库失败时回滚MinIO上传 (需求 2.8)"""
        db_manager, mock_session = mock_db_manager

        # Make commit raise an error
        async def failing_commit():
            raise Exception("Database commit failed")

        mock_session.commit = failing_commit

        from gateway.api.file_service import FileService

        file_service = FileService(minio_service=mock_minio_service)
        file_service._initialized = True

        app = create_test_app(override_user=authenticated_user)
        client = TestClient(app)

        with patch("gateway.api.routers.v1.file_service", file_service), \
             patch("gateway.api.file_service.get_database", new=AsyncMock(return_value=db_manager)):

            response = client.post(
                "/api/v1/files/upload",
                files={"file": ("test.pdf", b"content", "application/pdf")},
            )

        # Should fail
        assert response.status_code == 500

        # MinIO delete should have been called to rollback
        mock_minio_service.delete_file.assert_called_once()

    def test_download_file_requires_auth(self):
        """测试文件下载需要认证 (需求 6.7)"""
        app = FastAPI()

        async def mock_get_service_client():
            return None

        app.dependency_overrides[get_service_client] = mock_get_service_client
        app.include_router(v1_router, prefix="/api/v1")

        client = TestClient(app)

        response = client.get("/api/v1/files/test-file-id")
        assert response.status_code == 401

    def test_delete_file_requires_auth(self):
        """测试文件删除需要认证 (需求 6.7)"""
        app = FastAPI()

        async def mock_get_service_client():
            return None

        app.dependency_overrides[get_service_client] = mock_get_service_client
        app.include_router(v1_router, prefix="/api/v1")

        client = TestClient(app)

        response = client.delete("/api/v1/files/test-file-id")
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
