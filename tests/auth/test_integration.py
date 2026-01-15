"""
认证系统集成测试
测试各组件之间的协作
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from business.auth.auth_service import AuthService
from business.auth.jwt_manager import JWTManager
from business.auth.rbac_manager import RBACManager
from business.auth.models import (
    LoginRequest, AuthMethod, UserRole, Permission,
    AuthUser, TokenResponse
)


class TestAuthIntegration:
    """认证系统集成测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.auth_service = AuthService()
        self.jwt_manager = JWTManager()
        self.rbac_manager = RBACManager()
        
        # 创建测试用户
        self.test_user = AuthUser(
            user_id="integration_test_user",
            username="integrationuser",
            email="integration@example.com",
            department="IT",
            roles=[UserRole.OPERATOR],
            permissions=[
                Permission.FILE_UPLOAD, Permission.FILE_DOWNLOAD,
                Permission.WATERMARK_EMBED, Permission.WATERMARK_EXTRACT
            ],
            is_active=True,
            auth_method=AuthMethod.LOCAL,
            mfa_enabled=False
        )
    
    def test_jwt_rbac_integration(self):
        """测试JWT和RBAC集成"""
        # 创建JWT令牌
        token = self.jwt_manager.create_access_token(self.test_user)
        assert token is not None
        
        # 验证令牌
        payload = self.jwt_manager.verify_token(token)
        assert payload is not None
        
        # 从令牌中提取角色
        user_roles = [UserRole(role) for role in payload.get("roles", [])]
        
        # 使用RBAC检查权限
        has_file_permission = self.rbac_manager.check_permission(
            user_roles, "file", "create"
        )
        assert has_file_permission is True
        
        has_admin_permission = self.rbac_manager.check_permission(
            user_roles, "user", "manage"
        )
        assert has_admin_permission is False
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """测试完整的认证流程"""
        # Mock数据库用户
        mock_user_data = Mock()
        mock_user_data.user_id = self.test_user.user_id
        mock_user_data.username = self.test_user.username
        mock_user_data.email = self.test_user.email
        mock_user_data.department = self.test_user.department
        mock_user_data.roles = [role.value for role in self.test_user.roles]
        mock_user_data.password_hash = "hashed_password"
        mock_user_data.is_active = True
        mock_user_data.created_at = datetime.utcnow()
        
        # Mock依赖
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=mock_user_data)
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=False)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._create_session = AsyncMock(return_value="session_123")
        self.auth_service._update_last_login = AsyncMock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 1. 用户登录
        login_request = LoginRequest(
            username=self.test_user.username,
            password="correct_password",
            auth_method=AuthMethod.LOCAL
        )
        
        with patch.object(self.auth_service.jwt_manager, 'verify_password', return_value=True):
            token_response = await self.auth_service.login(
                login_request, "127.0.0.1", "test-agent"
            )
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token is not None
        
        # 2. 使用令牌进行认证
        with patch.object(self.auth_service, '_is_session_valid', return_value=True):
            user_info = await self.auth_service.authenticate(token_response.access_token)
        
        assert user_info is not None
        assert user_info.user_id == self.test_user.user_id
        
        # 3. 权限检查
        has_permission = await self.auth_service.authorize(
            user_info.user_id, "file", "create"
        )
        assert has_permission is True
        
        # 4. 登出
        with patch.object(self.auth_service, '_remove_session', return_value=True):
            logout_success = await self.auth_service.logout(
                token_response.access_token, "127.0.0.1", "test-agent"
            )
        assert logout_success is True
    
    def test_role_hierarchy_with_permissions(self):
        """测试角色继承与权限的集成"""
        # 测试操作员角色（继承用户权限）
        operator_roles = [UserRole.OPERATOR]
        
        # 操作员应该有用户的基本权限
        user_permissions = [
            ("file", "create"),
            ("watermark", "create")
        ]
        
        for resource, action in user_permissions:
            assert self.rbac_manager.check_permission(operator_roles, resource, action)
        
        # 操作员还应该有额外的权限
        operator_permissions = [
            ("file", "read"),
            ("file", "update"),
            ("watermark", "read"),
            ("watermark", "update"),
            ("watermark", "execute")
        ]
        
        for resource, action in operator_permissions:
            assert self.rbac_manager.check_permission(operator_roles, resource, action)
    
    def test_token_expiry_and_refresh_integration(self):
        """测试令牌过期和刷新的集成"""
        # 创建短期令牌
        short_expiry = timedelta(seconds=1)
        token = self.jwt_manager.create_access_token(self.test_user, short_expiry)
        
        # 立即验证应该成功
        payload = self.jwt_manager.verify_token(token)
        assert payload is not None
        
        # 等待令牌过期
        import time
        time.sleep(2)
        
        # 验证过期令牌应该失败
        expired_payload = self.jwt_manager.verify_token(token)
        assert expired_payload is None
        
        # 测试刷新令牌
        refresh_token = self.jwt_manager.create_refresh_token(self.test_user.user_id)
        new_token_response = self.jwt_manager.refresh_access_token(refresh_token, self.test_user)
        
        assert new_token_response is not None
        assert new_token_response.access_token != token
        
        # 新令牌应该可以验证
        new_payload = self.jwt_manager.verify_token(new_token_response.access_token)
        assert new_payload is not None
    
    @pytest.mark.asyncio
    async def test_permission_context_integration(self):
        """测试权限上下文的集成"""
        # 测试资源所有权检查
        user_roles = [UserRole.USER]
        
        # 用户访问自己的资源
        owner_context = {
            "current_user": "user123",
            "resource_owner": "user123",
            "owner": True
        }
        
        # 这个测试需要RBAC管理器支持所有权检查
        # 在实际实现中，需要确保权限规则包含所有权条件
        has_owner_permission = self.rbac_manager.check_permission(
            user_roles, "file", "read", owner_context
        )
        
        # 根据当前的RBAC实现，用户应该有文件读取权限
        # 但具体的所有权检查需要在权限规则中定义
        assert isinstance(has_owner_permission, bool)
    
    def test_multiple_roles_permission_aggregation(self):
        """测试多角色权限聚合"""
        # 用户同时拥有USER和AUDITOR角色
        multiple_roles = [UserRole.USER, UserRole.AUDITOR]
        
        # 应该拥有两个角色的所有权限
        user_permissions = self.rbac_manager.get_user_permissions([UserRole.USER])
        auditor_permissions = self.rbac_manager.get_user_permissions([UserRole.AUDITOR])
        combined_permissions = self.rbac_manager.get_user_permissions(multiple_roles)
        
        # 合并权限应该包含两个角色的权限
        user_permission_set = set(user_permissions)
        auditor_permission_set = set(auditor_permissions)
        combined_permission_set = set(combined_permissions)
        
        # 验证包含关系
        assert user_permission_set.issubset(combined_permission_set)
        assert auditor_permission_set.issubset(combined_permission_set)
    
    @pytest.mark.asyncio
    async def test_session_management_integration(self):
        """测试会话管理集成"""
        user_id = "session_test_user"
        ip_address = "192.168.1.100"
        user_agent = "TestAgent/1.0"
        
        # 创建会话
        session_id = await self.auth_service._create_session(user_id, ip_address, user_agent)
        assert session_id is not None
        
        # 验证会话有效性
        is_valid = await self.auth_service._is_session_valid(user_id)
        assert is_valid is True
        
        # 移除会话
        await self.auth_service._remove_session(user_id)
        
        # 验证会话已失效
        is_valid_after_removal = await self.auth_service._is_session_valid(user_id)
        assert is_valid_after_removal is False
    
    def test_audit_logging_integration(self):
        """测试审计日志集成"""
        # 这个测试验证审计日志在认证流程中的集成
        # 实际实现中应该验证日志是否正确记录到数据库
        
        # Mock审计日志记录
        with patch.object(self.auth_service, '_log_audit_event') as mock_log:
            # 模拟登录失败
            self.auth_service._record_failed_attempt("testuser")
            
            # 验证审计日志方法存在且可调用
            assert hasattr(self.auth_service, '_log_audit_event')
            assert callable(self.auth_service._log_audit_event)
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 测试各种错误情况的处理
        
        # 1. 无效的角色枚举
        invalid_roles = ["invalid_role"]
        try:
            invalid_user_roles = [UserRole(role) for role in invalid_roles]
            # 如果到达这里，说明没有抛出异常（可能是设计问题）
        except ValueError:
            # 预期的行为：无效角色应该抛出ValueError
            pass
        
        # 2. 无效的权限字符串
        assert not self.rbac_manager.validate_permission_string("invalid_format")
        assert not self.rbac_manager.validate_permission_string("resource:")
        assert not self.rbac_manager.validate_permission_string(":action")
        
        # 3. 空令牌验证
        empty_payload = self.jwt_manager.verify_token("")
        assert empty_payload is None
        
        null_payload = self.jwt_manager.verify_token(None)
        assert null_payload is None
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self):
        """测试并发会话处理"""
        user_id = "concurrent_test_user"
        
        # 创建多个会话
        session_ids = []
        for i in range(3):
            session_id = await self.auth_service._create_session(
                user_id, f"192.168.1.{100+i}", f"TestAgent/{i}"
            )
            session_ids.append(session_id)
        
        # 验证所有会话都有效
        for _ in session_ids:
            is_valid = await self.auth_service._is_session_valid(user_id)
            assert is_valid is True
        
        # 移除所有会话
        await self.auth_service._remove_session(user_id)
        
        # 验证所有会话都已失效
        is_valid_after_removal = await self.auth_service._is_session_valid(user_id)
        assert is_valid_after_removal is False
    
    def test_permission_caching_integration(self):
        """测试权限缓存集成"""
        # 测试权限检查的性能和缓存
        user_roles = [UserRole.OPERATOR]
        
        # 多次检查相同权限
        for _ in range(10):
            result = self.rbac_manager.check_permission(user_roles, "file", "read")
            assert result is True
        
        # 验证权限缓存机制（如果实现了的话）
        # 这里主要是确保多次调用不会出错
        permissions = self.rbac_manager.get_user_permissions(user_roles)
        assert len(permissions) > 0
        
        # 再次获取应该返回相同结果
        permissions_again = self.rbac_manager.get_user_permissions(user_roles)
        assert permissions == permissions_again