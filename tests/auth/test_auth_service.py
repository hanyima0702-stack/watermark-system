"""
认证服务测试
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from business.auth.auth_service import AuthService
from business.auth.models import (
    LoginRequest, AuthMethod, UserRole, Permission, 
    AuthUser, TokenResponse, MFAChallenge, MFAMethod
)
from shared.interfaces import UserInfo


class TestAuthService:
    """认证服务测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.auth_service = AuthService()
        
        # Mock依赖
        self.auth_service.user_dao = Mock()
        self.auth_service.jwt_manager = Mock()
        self.auth_service.ldap_client = Mock()
        self.auth_service.mfa_service = Mock()
        
        # 测试用户数据
        self.test_user_data = Mock()
        self.test_user_data.user_id = "test_user_001"
        self.test_user_data.username = "testuser"
        self.test_user_data.email = "test@example.com"
        self.test_user_data.department = "IT"
        self.test_user_data.roles = ["user", "operator"]
        self.test_user_data.password_hash = "hashed_password"
        self.test_user_data.is_active = True
        self.test_user_data.created_at = datetime.utcnow()
        
        self.test_auth_user = AuthUser(
            user_id="test_user_001",
            username="testuser",
            email="test@example.com",
            department="IT",
            roles=[UserRole.USER, UserRole.OPERATOR],
            permissions=[Permission.FILE_UPLOAD, Permission.WATERMARK_EMBED],
            is_active=True,
            auth_method=AuthMethod.LOCAL,
            mfa_enabled=False
        )
    
    @pytest.mark.asyncio
    async def test_login_success_local_auth(self):
        """测试本地认证登录成功"""
        # 准备测试数据
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LOCAL
        )
        
        # Mock依赖方法
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.jwt_manager.create_access_token = Mock(return_value="access_token")
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=False)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._create_session = AsyncMock(return_value="session_id")
        self.auth_service._update_last_login = AsyncMock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录
        result = await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
        
        # 验证结果
        assert isinstance(result, TokenResponse)
        assert result.access_token == "access_token"
        assert result.user_info.user_id == "test_user_001"
        assert not result.mfa_required
        
        # 验证调用
        self.auth_service.user_dao.get_by_username.assert_called_once_with("testuser")
        self.auth_service.jwt_manager.verify_password.assert_called_once()
        self.auth_service.jwt_manager.create_access_token.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self):
        """测试无效凭据登录"""
        login_request = LoginRequest(
            username="testuser",
            password="wrong_password",
            auth_method=AuthMethod.LOCAL
        )
        
        # Mock返回用户但密码验证失败
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=False)
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录并期望异常
        with pytest.raises(Exception, match="用户名或密码错误"):
            await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
    
    @pytest.mark.asyncio
    async def test_login_user_not_found(self):
        """测试用户不存在"""
        login_request = LoginRequest(
            username="nonexistent",
            password="password123",
            auth_method=AuthMethod.LOCAL
        )
        
        # Mock返回None（用户不存在）
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=None)
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录并期望异常
        with pytest.raises(Exception, match="用户名或密码错误"):
            await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
    
    @pytest.mark.asyncio
    async def test_login_inactive_user(self):
        """测试非活跃用户登录"""
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LOCAL
        )
        
        # 设置用户为非活跃状态
        inactive_user_data = Mock()
        inactive_user_data.user_id = "test_user_001"
        inactive_user_data.username = "testuser"
        inactive_user_data.email = "test@example.com"
        inactive_user_data.department = "IT"
        inactive_user_data.roles = ["user"]
        inactive_user_data.password_hash = "hashed_password"
        inactive_user_data.is_active = False
        
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=inactive_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=False)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录并期望异常
        with pytest.raises(Exception, match="账户已被禁用"):
            await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
    
    @pytest.mark.asyncio
    async def test_login_with_mfa_required(self):
        """测试需要MFA的登录"""
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LOCAL
        )
        
        mfa_challenge = MFAChallenge(
            challenge_id="challenge_123",
            method=MFAMethod.TOTP,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        # Mock MFA启用
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=True)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._initiate_mfa_challenge = AsyncMock(return_value=mfa_challenge)
        self.auth_service._clear_failed_attempts = Mock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录
        result = await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
        
        # 验证MFA挑战
        assert result.mfa_required is True
        assert result.mfa_challenge == mfa_challenge
        assert result.access_token == ""
    
    @pytest.mark.asyncio
    async def test_login_with_mfa_verification(self):
        """测试MFA验证登录"""
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LOCAL,
            mfa_code="123456"
        )
        
        # Mock MFA验证成功
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.jwt_manager.create_access_token = Mock(return_value="access_token")
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=True)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._verify_mfa = AsyncMock(return_value=True)
        self.auth_service._create_session = AsyncMock(return_value="session_id")
        self.auth_service._update_last_login = AsyncMock()
        self.auth_service._clear_failed_attempts = Mock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录
        result = await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
        
        # 验证结果
        assert isinstance(result, TokenResponse)
        assert result.access_token == "access_token"
        assert not result.mfa_required
        
        # 验证MFA验证被调用
        self.auth_service._verify_mfa.assert_called_once_with("test_user_001", "123456")
    
    @pytest.mark.asyncio
    async def test_login_mfa_verification_failed(self):
        """测试MFA验证失败"""
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LOCAL,
            mfa_code="wrong_code"
        )
        
        # Mock MFA验证失败
        self.auth_service.user_dao.get_by_username = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.mfa_service.is_mfa_enabled = Mock(return_value=True)
        self.auth_service._get_user_permissions = AsyncMock(return_value=[])
        self.auth_service._verify_mfa = AsyncMock(return_value=False)
        self.auth_service._clear_failed_attempts = Mock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录并期望异常
        with pytest.raises(Exception, match="MFA验证失败"):
            await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
    
    @pytest.mark.asyncio
    async def test_login_ldap_auth(self):
        """测试LDAP认证登录"""
        login_request = LoginRequest(
            username="testuser",
            password="password123",
            auth_method=AuthMethod.LDAP
        )
        
        # Mock LDAP认证成功
        self.auth_service.ldap_client.authenticate = Mock(return_value=self.test_auth_user)
        self.auth_service.jwt_manager.create_access_token = Mock(return_value="access_token")
        self.auth_service._create_session = AsyncMock(return_value="session_id")
        self.auth_service._update_last_login = AsyncMock()
        self.auth_service._clear_failed_attempts = Mock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登录
        result = await self.auth_service.login(login_request, "127.0.0.1", "test-agent")
        
        # 验证结果
        assert isinstance(result, TokenResponse)
        assert result.access_token == "access_token"
        assert result.user_info.auth_method == AuthMethod.LDAP
        
        # 验证LDAP认证被调用
        self.auth_service.ldap_client.authenticate.assert_called_once_with("testuser", "password123")
    
    @pytest.mark.asyncio
    async def test_logout_success(self):
        """测试登出成功"""
        token = "valid_token"
        
        # Mock令牌验证和撤销
        mock_payload = {"sub": "test_user_001"}
        self.auth_service.jwt_manager.verify_token = Mock(return_value=mock_payload)
        self.auth_service.jwt_manager.revoke_token = Mock(return_value=True)
        self.auth_service._remove_session = AsyncMock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行登出
        result = await self.auth_service.logout(token, "127.0.0.1", "test-agent")
        
        # 验证结果
        assert result is True
        
        # 验证调用
        self.auth_service.jwt_manager.verify_token.assert_called_once_with(token)
        self.auth_service.jwt_manager.revoke_token.assert_called_once_with(token)
        self.auth_service._remove_session.assert_called_once_with("test_user_001")
    
    @pytest.mark.asyncio
    async def test_logout_invalid_token(self):
        """测试无效令牌登出"""
        token = "invalid_token"
        
        # Mock令牌验证失败
        self.auth_service.jwt_manager.verify_token = Mock(return_value=None)
        
        # 执行登出
        result = await self.auth_service.logout(token, "127.0.0.1", "test-agent")
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self):
        """测试有效令牌认证"""
        token = "valid_token"
        
        # Mock令牌验证成功
        mock_payload = {
            "sub": "test_user_001",
            "username": "testuser",
            "email": "test@example.com",
            "department": "IT",
            "roles": ["user", "operator"]
        }
        
        self.auth_service.jwt_manager.verify_token = Mock(return_value=mock_payload)
        self.auth_service.jwt_manager.is_token_blacklisted = Mock(return_value=False)
        self.auth_service._is_session_valid = AsyncMock(return_value=True)
        
        # 执行认证
        result = await self.auth_service.authenticate(token)
        
        # 验证结果
        assert isinstance(result, UserInfo)
        assert result.user_id == "test_user_001"
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        assert result.roles == ["user", "operator"]
    
    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self):
        """测试无效令牌认证"""
        token = "invalid_token"
        
        # Mock令牌验证失败
        self.auth_service.jwt_manager.verify_token = Mock(return_value=None)
        
        # 执行认证
        result = await self.auth_service.authenticate(token)
        
        # 验证结果
        assert result is None
    
    @pytest.mark.asyncio
    async def test_authenticate_blacklisted_token(self):
        """测试黑名单令牌认证"""
        token = "blacklisted_token"
        
        # Mock令牌在黑名单中
        mock_payload = {"sub": "test_user_001"}
        self.auth_service.jwt_manager.verify_token = Mock(return_value=mock_payload)
        self.auth_service.jwt_manager.is_token_blacklisted = Mock(return_value=True)
        
        # 执行认证
        result = await self.auth_service.authenticate(token)
        
        # 验证结果
        assert result is None
    
    @pytest.mark.asyncio
    async def test_authorize_admin_user(self):
        """测试管理员用户授权"""
        user_id = "admin_user"
        
        # Mock管理员用户信息
        admin_user_info = UserInfo(
            user_id=user_id,
            username="admin",
            email="admin@example.com",
            department="IT",
            roles=["admin"],
            created_at=datetime.utcnow()
        )
        
        self.auth_service.get_user_info = AsyncMock(return_value=admin_user_info)
        
        # 执行授权检查
        result = await self.auth_service.authorize(user_id, "file", "delete")
        
        # 管理员应该有所有权限
        assert result is True
    
    @pytest.mark.asyncio
    async def test_authorize_regular_user(self):
        """测试普通用户授权"""
        user_id = "regular_user"
        
        # Mock普通用户信息
        regular_user_info = UserInfo(
            user_id=user_id,
            username="user",
            email="user@example.com",
            department="IT",
            roles=["user"],
            created_at=datetime.utcnow()
        )
        
        self.auth_service.get_user_info = AsyncMock(return_value=regular_user_info)
        self.auth_service._check_permission = AsyncMock(return_value=True)
        
        # 执行授权检查
        result = await self.auth_service.authorize(user_id, "file", "read")
        
        # 验证结果
        assert result is True
        self.auth_service._check_permission.assert_called_once_with(user_id, "file:read")
    
    @pytest.mark.asyncio
    async def test_get_user_info_success(self):
        """测试获取用户信息成功"""
        user_id = "test_user_001"
        
        # Mock用户数据
        self.auth_service.user_dao.get_by_id = AsyncMock(return_value=self.test_user_data)
        
        # 执行获取用户信息
        result = await self.auth_service.get_user_info(user_id)
        
        # 验证结果
        assert isinstance(result, UserInfo)
        assert result.user_id == user_id
        assert result.username == "testuser"
        assert result.email == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self):
        """测试用户不存在"""
        user_id = "nonexistent_user"
        
        # Mock用户不存在
        self.auth_service.user_dao.get_by_id = AsyncMock(return_value=None)
        
        # 执行获取用户信息
        result = await self.auth_service.get_user_info(user_id)
        
        # 验证结果
        assert result is None
    
    @pytest.mark.asyncio
    async def test_change_password_success(self):
        """测试修改密码成功"""
        user_id = "test_user_001"
        current_password = "old_password"
        new_password = "new_password"
        
        # Mock密码验证和更新
        self.auth_service.user_dao.get_by_id = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=True)
        self.auth_service.jwt_manager.hash_password = Mock(return_value="new_hashed_password")
        self.auth_service.user_dao.update_password = AsyncMock()
        self.auth_service._log_audit_event = AsyncMock()
        
        # 执行修改密码
        result = await self.auth_service.change_password(user_id, current_password, new_password)
        
        # 验证结果
        assert result is True
        
        # 验证调用
        self.auth_service.jwt_manager.verify_password.assert_called_once_with(
            current_password, self.test_user_data.password_hash
        )
        self.auth_service.user_dao.update_password.assert_called_once_with(
            user_id, "new_hashed_password"
        )
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current_password(self):
        """测试当前密码错误"""
        user_id = "test_user_001"
        current_password = "wrong_password"
        new_password = "new_password"
        
        # Mock当前密码验证失败
        self.auth_service.user_dao.get_by_id = AsyncMock(return_value=self.test_user_data)
        self.auth_service.jwt_manager.verify_password = Mock(return_value=False)
        
        # 执行修改密码
        result = await self.auth_service.change_password(user_id, current_password, new_password)
        
        # 验证结果
        assert result is False
    
    def test_account_lockout_mechanism(self):
        """测试账户锁定机制"""
        username = "testuser"
        
        # 测试初始状态不锁定
        assert not self.auth_service._is_account_locked(username)
        
        # 记录多次失败尝试
        for _ in range(5):
            self.auth_service._record_failed_attempt(username)
        
        # 验证账户被锁定
        assert self.auth_service._is_account_locked(username)
        
        # 清除失败尝试
        self.auth_service._clear_failed_attempts(username)
        
        # 验证锁定仍然有效（因为锁定时间未过）
        assert self.auth_service._is_account_locked(username)