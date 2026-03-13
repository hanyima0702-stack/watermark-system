"""
Watermark API Authentication Integration Tests
测试水印嵌入和提取API的认证集成

Run with: python -m pytest tests/test_watermark_api_auth.py -v
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api.routers.v1 import router as v1_router
from gateway.api.dependencies import get_current_user, get_service_client


# ============= Helpers =============

def create_test_app(override_user=None):
    """Create a test FastAPI app with optional auth override."""
    app = FastAPI()

    if override_user is not None:
        async def mock_get_current_user():
            return override_user
        app.dependency_overrides[get_current_user] = mock_get_current_user

    async def mock_get_service_client():
        return None
    app.dependency_overrides[get_service_client] = mock_get_service_client

    app.include_router(v1_router, prefix="/api/v1")
    return app


@pytest.fixture
def authenticated_user():
    """Authenticated user data."""
    return {"user_id": "user_abc123", "username": "testuser"}


@pytest.fixture
def auth_client(authenticated_user):
    """Test client with authenticated user."""
    app = create_test_app(override_user=authenticated_user)
    return TestClient(app)


@pytest.fixture
def no_auth_client():
    """Test client without authentication override — dependency raises 401."""
    app = FastAPI()

    async def mock_get_service_client():
        return None
    app.dependency_overrides[get_service_client] = mock_get_service_client

    app.include_router(v1_router, prefix="/api/v1")
    return TestClient(app)


# ============= 水印嵌入API认证测试 =============

class TestWatermarkEmbedAuth:
    """测试带认证的水印嵌入API"""

    def test_embed_with_valid_auth(self, auth_client):
        """测试已认证用户可以创建水印嵌入任务"""
        response = auth_client.post(
            "/api/v1/watermark/embed",
            json={
                "file_id": "file_001",
                "config_id": "config_001",
                "watermark_type": "both",
                "priority": 0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_123"
        assert data["status"] == "pending"
        assert "user_abc123" in data["message"]

    def test_embed_without_auth_returns_401(self, no_auth_client):
        """测试未认证用户创建水印嵌入任务被拒绝"""
        response = no_auth_client.post(
            "/api/v1/watermark/embed",
            json={
                "file_id": "file_001",
                "config_id": "config_001",
                "watermark_type": "both",
            },
        )
        assert response.status_code == 401

    def test_embed_task_associated_with_user(self, auth_client):
        """测试水印嵌入任务关联到当前用户"""
        response = auth_client.post(
            "/api/v1/watermark/embed",
            json={
                "file_id": "file_002",
                "config_id": "config_002",
                "watermark_type": "visible",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_abc123" in data["message"]

    def test_get_task_status_with_auth(self, auth_client):
        """测试已认证用户可以查询任务状态"""
        response = auth_client.get("/api/v1/watermark/tasks/task_123")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_123"
        assert data["status"] == "processing"

    def test_get_task_status_without_auth(self, no_auth_client):
        """测试未认证用户查询任务状态被拒绝"""
        response = no_auth_client.get("/api/v1/watermark/tasks/task_123")
        assert response.status_code == 401

    def test_cancel_task_with_auth(self, auth_client):
        """测试已认证用户可以取消任务"""
        response = auth_client.post("/api/v1/watermark/tasks/task_123/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Task cancelled successfully"

    def test_cancel_task_without_auth(self, no_auth_client):
        """测试未认证用户取消任务被拒绝"""
        response = no_auth_client.post("/api/v1/watermark/tasks/task_123/cancel")
        assert response.status_code == 401

    def test_list_tasks_with_auth(self, auth_client):
        """测试已认证用户可以列出任务"""
        response = auth_client.get("/api/v1/watermark/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_tasks_without_auth(self, no_auth_client):
        """测试未认证用户列出任务被拒绝"""
        response = no_auth_client.get("/api/v1/watermark/tasks")
        assert response.status_code == 401


# ============= 水印提取API认证测试 =============

class TestWatermarkExtractAuth:
    """测试带认证的水印提取API"""

    def test_extract_with_valid_auth(self, auth_client):
        """测试已认证用户可以提取水印"""
        response = auth_client.post(
            "/api/v1/watermark/extract",
            json={"file_id": "file_001", "generate_report": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == "result_123"
        assert data["file_id"] == "file_001"
        assert "confidence_score" in data

    def test_extract_without_auth_returns_401(self, no_auth_client):
        """测试未认证用户提取水印被拒绝"""
        response = no_auth_client.post(
            "/api/v1/watermark/extract",
            json={"file_id": "file_001", "generate_report": True},
        )
        assert response.status_code == 401

    def test_extract_with_methods(self, auth_client):
        """测试指定提取方法的水印提取"""
        response = auth_client.post(
            "/api/v1/watermark/extract",
            json={
                "file_id": "file_003",
                "extraction_methods": ["dct", "dwt"],
                "generate_report": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["file_id"] == "file_003"


# ============= 不同用户隔离测试 =============

class TestUserIsolation:
    """测试不同用户之间的任务隔离"""

    def test_different_users_get_different_responses(self):
        """测试不同用户创建的任务关联到各自的用户ID"""
        user_a = {"user_id": "user_aaa", "username": "alice"}
        user_b = {"user_id": "user_bbb", "username": "bob"}

        client_a = TestClient(create_test_app(override_user=user_a))
        client_b = TestClient(create_test_app(override_user=user_b))

        resp_a = client_a.post(
            "/api/v1/watermark/embed",
            json={"file_id": "f1", "config_id": "c1", "watermark_type": "visible"},
        )
        resp_b = client_b.post(
            "/api/v1/watermark/embed",
            json={"file_id": "f2", "config_id": "c2", "watermark_type": "invisible"},
        )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        assert "user_aaa" in resp_a.json()["message"]
        assert "user_bbb" in resp_b.json()["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
