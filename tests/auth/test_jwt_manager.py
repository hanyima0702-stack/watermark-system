"""
JWT管理器测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from business.auth.jwt_manager import JWTManager
from business.auth.models import AuthUser, UserRole, Permission, AuthMethod


class TestJWTManager:
    """JWT管理器测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.jwt_manager = JWTManager()
        self.test_user = AuthUser(
            user_id="test_user_001",
            username="testuser",
            email="test@example.com",
            department="IT",
            roles=[UserRole.USER, UserRole.OPERATOR],
            permissions=[Permission.FILE_UPLOAD, Permission.WATERMARK_EMBED],
            is_active=True,
            auth_method=AuthMethod.LOCAL
        )
    
    def test_create_access_token(self):
        """测试创建访问令牌"""
        token = self.jwt_manager.create_access_token(self.test_user)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_access_token_with_custom_expiry(self):
        """测试创建带自定义过期时间的访问令牌"""
        custom_expiry = timedelta(minutes=30)
        token = self.jwt_manager.create_access_token(self.test_user, custom_expiry)
        
        # 验证令牌
        payload = self.jwt_manager.verify_token(token)
        assert payload is not None
        
        # 检查过期时间
        exp_timestamp = payload.get("exp")
        exp_datetime = datetime.fromtimestamp(exp_timestamp)
        expected_exp = datetime.utcnow() + custom_expiry
        
        # 允许1分钟的误差
        assert abs((exp_datetime - expected_exp).total_seconds()) < 60
    
    def test_create_refresh_token(self):
        """测试创建刷新令牌"""
        refresh_token = self.jwt_manager.create_refresh_token(self.test_user.user_id)
        
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        assert len(refresh_token) > 0
    
    def test_verify_valid_token(self):
        """测试验证有效令牌"""
        token = self.jwt_manager.create_access_token(self.test_user)
        payload = self.jwt_manager.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == self.test_user.user_id
        assert payload["username"] == self.test_user.username
        assert payload["email"] == self.test_user.email
        assert payload["type"] == "access"
    
    def test_verify_invalid_token(self):
        """测试验证无效令牌"""
        invalid_token = "invalid.token.here"
        payload = self.jwt_manager.verify_token(invalid_token)
        
        assert payload is None
    
    def test_verify_expired_token(self):
        """测试验证过期令牌"""
        # 创建一个已过期的令牌
        expired_expiry = timedelta(seconds=-1)
        token = self.jwt_manager.create_access_token(self.test_user, expired_expiry)
        
        # 等待令牌过期
        import time
        time.sleep(2)
        
        payload = self.jwt_manager.verify_token(token)
        assert payload is None
    
    def test_verify_refresh_token(self):
        """测试验证刷新令牌"""
        refresh_token = self.jwt_manager.create_refresh_token(self.test_user.user_id)
        user_id = self.jwt_manager.verify_refresh_token(refresh_token)
        
        assert user_id == self.test_user.user_id
    
    def test_verify_invalid_refresh_token(self):
        """测试验证无效刷新令牌"""
        invalid_refresh_token = "invalid.refresh.token"
        user_id = self.jwt_manager.verify_refresh_token(invalid_refresh_token)
        
        assert user_id is None
    
    def test_refresh_access_token(self):
        """测试刷新访问令牌"""
        refresh_token = self.jwt_manager.create_refresh_token(self.test_user.user_id)
        token_response = self.jwt_manager.refresh_access_token(refresh_token, self.test_user)
        
        assert token_response is not None
        assert token_response.access_token is not None
        assert token_response.refresh_token == refresh_token
        assert token_response.user_info.user_id == self.test_user.user_id
    
    def test_refresh_access_token_with_invalid_refresh_token(self):
        """测试使用无效刷新令牌刷新访问令牌"""
        invalid_refresh_token = "invalid.refresh.token"
        token_response = self.jwt_manager.refresh_access_token(invalid_refresh_token, self.test_user)
        
        assert token_response is None
    
    def test_decode_token_payload(self):
        """测试解码令牌载荷"""
        token = self.jwt_manager.create_access_token(self.test_user)
        payload = self.jwt_manager.decode_token_payload(token)
        
        assert payload is not None
        assert payload["sub"] == self.test_user.user_id
        assert payload["username"] == self.test_user.username
    
    def test_get_token_expiry(self):
        """测试获取令牌过期时间"""
        token = self.jwt_manager.create_access_token(self.test_user)
        expiry = self.jwt_manager.get_token_expiry(token)
        
        assert expiry is not None
        assert isinstance(expiry, datetime)
        assert expiry > datetime.utcnow()
    
    def test_is_token_expired(self):
        """测试检查令牌是否过期"""
        # 测试未过期的令牌
        token = self.jwt_manager.create_access_token(self.test_user)
        assert not self.jwt_manager.is_token_expired(token)
        
        # 测试过期的令牌
        expired_token = self.jwt_manager.create_access_token(
            self.test_user, 
            timedelta(seconds=-1)
        )
        import time
        time.sleep(2)
        assert self.jwt_manager.is_token_expired(expired_token)
    
    def test_hash_password(self):
        """测试密码哈希"""
        password = "test_password_123"
        hashed = self.jwt_manager.hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """测试密码验证"""
        password = "test_password_123"
        hashed = self.jwt_manager.hash_password(password)
        
        # 验证正确密码
        assert self.jwt_manager.verify_password(password, hashed)
        
        # 验证错误密码
        assert not self.jwt_manager.verify_password("wrong_password", hashed)
    
    def test_generate_password_reset_token(self):
        """测试生成密码重置令牌"""
        reset_token = self.jwt_manager.generate_password_reset_token(self.test_user.user_id)
        
        assert reset_token is not None
        assert isinstance(reset_token, str)
        assert len(reset_token) > 0
    
    def test_verify_password_reset_token(self):
        """测试验证密码重置令牌"""
        reset_token = self.jwt_manager.generate_password_reset_token(self.test_user.user_id)
        user_id = self.jwt_manager.verify_password_reset_token(reset_token)
        
        assert user_id == self.test_user.user_id
    
    def test_verify_invalid_password_reset_token(self):
        """测试验证无效密码重置令牌"""
        invalid_token = "invalid.reset.token"
        user_id = self.jwt_manager.verify_password_reset_token(invalid_token)
        
        assert user_id is None
    
    @patch('business.auth.jwt_manager.jose_jwt.decode')
    def test_verify_token_with_jwt_error(self, mock_decode):
        """测试JWT解码错误处理"""
        from jose import JWTError
        mock_decode.side_effect = JWTError("Invalid token")
        
        token = "some.invalid.token"
        payload = self.jwt_manager.verify_token(token)
        
        assert payload is None
    
    def test_token_payload_structure(self):
        """测试令牌载荷结构"""
        token = self.jwt_manager.create_access_token(self.test_user)
        payload = self.jwt_manager.decode_token_payload(token)
        
        # 检查必需字段
        required_fields = ["sub", "username", "email", "roles", "permissions", "exp", "iat", "jti", "type"]
        for field in required_fields:
            assert field in payload
        
        # 检查字段值
        assert payload["sub"] == self.test_user.user_id
        assert payload["username"] == self.test_user.username
        assert payload["email"] == self.test_user.email
        assert payload["department"] == self.test_user.department
        assert payload["type"] == "access"
        
        # 检查角色和权限
        expected_roles = [role.value for role in self.test_user.roles]
        expected_permissions = [perm.value for perm in self.test_user.permissions]
        assert payload["roles"] == expected_roles
        assert payload["permissions"] == expected_permissions