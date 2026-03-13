"""
API Gateway Unit Tests
测试API网关的核心功能
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import time

from gateway.api.main import app
from gateway.api.config import settings


@pytest.fixture
def client():
    """测试客户端"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """认证头"""
    return {"Authorization": "Bearer test_token"}


class TestHealthEndpoints:
    """健康检查端点测试"""
    
    def test_root_endpoint(self, client):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Watermark System API Gateway"
        assert "version" in data
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
    
    def test_readiness_check(self, client):
        """测试就绪检查"""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "services" in data
    
    def test_liveness_check(self, client):
        """测试存活检查"""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestAuthentication:
    """认证测试"""
    
    def test_public_endpoints_no_auth(self, client):
        """测试公开端点不需要认证"""
        public_endpoints = ["/", "/health/", "/docs", "/openapi.json"]
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 404]  # 404 for non-existent endpoints
    
    def test_protected_endpoints_require_auth(self, client):
        """测试受保护端点需要认证"""
        response = client.get("/api/v1/watermark/tasks")
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert data["error"] == "Unauthorized"
    
    def test_invalid_token_format(self, client):
        """测试无效的token格式"""
        response = client.get(
            "/api/v1/watermark/tasks",
            headers={"Authorization": "InvalidFormat"}
        )
        assert response.status_code == 401
    
    def test_valid_token(self, client, auth_headers):
        """测试有效token"""
        response = client.get("/api/v1/watermark/tasks", headers=auth_headers)
        # 应该通过认证，但可能返回其他错误（如未实现）
        assert response.status_code != 401


class TestRateLimiting:
    """限流测试"""
    
    @pytest.mark.skipif(
        not settings.RATE_LIMIT_ENABLED,
        reason="Rate limiting disabled"
    )
    def test_rate_limit_exceeded(self, client, auth_headers):
        """测试超过限流阈值"""
        # 发送大量请求
        responses = []
        for _ in range(settings.RATE_LIMIT_REQUESTS + 10):
            response = client.get("/api/v1/watermark/tasks", headers=auth_headers)
            responses.append(response)
        
        # 检查是否有429响应
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes


class TestAPIVersioning:
    """API版本管理测试"""
    
    def test_v1_endpoints_exist(self, client, auth_headers):
        """测试v1端点存在"""
        response = client.get("/api/v1/watermark/tasks", headers=auth_headers)
        assert response.status_code != 404
    
    def test_v2_endpoints_exist(self, client, auth_headers):
        """测试v2端点存在"""
        response = client.get("/api/v2/watermark/tasks", headers=auth_headers)
        assert response.status_code != 404
    
    def test_v2_backward_compatible(self, client, auth_headers):
        """测试v2向后兼容v1"""
        # v2应该包含v1的所有端点
        v1_response = client.get("/api/v1/watermark/tasks", headers=auth_headers)
        v2_response = client.get("/api/v2/watermark/tasks", headers=auth_headers)
        
        # 两者应该返回相同的状态码（除了404）
        if v1_response.status_code != 404:
            assert v2_response.status_code == v1_response.status_code


class TestRequestLogging:
    """请求日志测试"""
    
    def test_request_id_in_response(self, client):
        """测试响应中包含请求ID"""
        response = client.get("/")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestMetrics:
    """监控指标测试"""
    
    def test_metrics_endpoint_exists(self, client):
        """测试metrics端点存在"""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_format(self, client):
        """测试metrics格式"""
        response = client.get("/metrics")
        content = response.text
        
        # Prometheus格式检查
        assert "api_requests_total" in content or "# HELP" in content


class TestErrorHandling:
    """错误处理测试"""
    
    def test_404_error(self, client):
        """测试404错误"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """测试方法不允许"""
        response = client.post("/")
        assert response.status_code == 405


class TestCORS:
    """CORS测试"""
    
    def test_cors_headers(self, client):
        """测试CORS头"""
        response = client.options(
            "/api/v1/watermark/tasks",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # 检查CORS相关头
        assert "access-control-allow-origin" in response.headers or response.status_code == 200


@pytest.mark.asyncio
class TestServiceClient:
    """服务客户端测试"""
    
    async def test_service_client_creation(self):
        """测试服务客户端创建"""
        from gateway.api.dependencies import ServiceClient
        
        client = ServiceClient()
        assert client.business_url == settings.BUSINESS_SERVICE_URL
        assert client.storage_url == settings.STORAGE_SERVICE_URL
        assert client.engine_url == settings.ENGINE_SERVICE_URL
    
    @patch('httpx.AsyncClient')
    async def test_call_business_service(self, mock_client):
        """测试调用业务服务"""
        from gateway.api.dependencies import ServiceClient
        
        # Mock响应
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = Mock()
        
        mock_client_instance = AsyncMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        client = ServiceClient()
        result = await client.call_business_service("GET", "/test")
        
        assert result == {"result": "success"}
        mock_client_instance.request.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
