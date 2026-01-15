"""
JWT令牌管理器
负责JWT令牌的生成、验证和刷新
"""

import jwt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt as jose_jwt
from passlib.context import CryptContext

from ...shared.config import get_settings
from .models import AuthUser, TokenResponse


class JWTManager:
    """JWT令牌管理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.security.jwt_secret_key
        self.algorithm = self.settings.security.jwt_algorithm
        self.access_token_expire_minutes = self.settings.security.jwt_expire_minutes
        self.refresh_token_expire_days = 7
        
        # 密码加密上下文
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(
        self, 
        user: AuthUser, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # 构建JWT载荷
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "department": user.department,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # JWT ID
            "type": "access"
        }
        
        return jose_jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """创建刷新令牌"""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        }
        
        return jose_jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证令牌"""
        try:
            payload = jose_jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 检查令牌类型
            if payload.get("type") != "access":
                return None
            
            # 检查过期时间
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                return None
            
            return payload
            
        except JWTError:
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[str]:
        """验证刷新令牌并返回用户ID"""
        try:
            payload = jose_jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 检查令牌类型
            if payload.get("type") != "refresh":
                return None
            
            # 检查过期时间
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                return None
            
            return payload.get("sub")
            
        except JWTError:
            return None
    
    def refresh_access_token(self, refresh_token: str, user: AuthUser) -> Optional[TokenResponse]:
        """使用刷新令牌生成新的访问令牌"""
        user_id = self.verify_refresh_token(refresh_token)
        if not user_id or user_id != user.user_id:
            return None
        
        # 生成新的访问令牌
        access_token = self.create_access_token(user)
        
        return TokenResponse(
            access_token=access_token,
            expires_in=self.access_token_expire_minutes * 60,
            refresh_token=refresh_token,  # 保持原有刷新令牌
            user_info=user
        )
    
    def decode_token_payload(self, token: str) -> Optional[Dict[str, Any]]:
        """解码令牌载荷（不验证签名，用于调试）"""
        try:
            return jose_jwt.get_unverified_claims(token)
        except JWTError:
            return None
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """获取令牌过期时间"""
        payload = self.decode_token_payload(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        return None
    
    def is_token_expired(self, token: str) -> bool:
        """检查令牌是否已过期"""
        expiry = self.get_token_expiry(token)
        if expiry:
            return datetime.utcnow() > expiry
        return True
    
    def revoke_token(self, token: str) -> bool:
        """撤销令牌（添加到黑名单）"""
        # 这里应该将令牌添加到Redis黑名单中
        # 实际实现需要配合Redis缓存
        payload = self.decode_token_payload(token)
        if payload and "jti" in payload:
            # TODO: 将JTI添加到Redis黑名单
            # redis_client.setex(f"blacklist:{payload['jti']}", expiry_seconds, "revoked")
            return True
        return False
    
    def is_token_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        payload = self.decode_token_payload(token)
        if payload and "jti" in payload:
            # TODO: 检查Redis黑名单
            # return redis_client.exists(f"blacklist:{payload['jti']}")
            return False
        return True
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_password_reset_token(self, user_id: str) -> str:
        """生成密码重置令牌"""
        expire = datetime.utcnow() + timedelta(hours=1)  # 1小时有效期
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "password_reset"
        }
        
        return jose_jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_password_reset_token(self, token: str) -> Optional[str]:
        """验证密码重置令牌"""
        try:
            payload = jose_jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "password_reset":
                return None
            
            return payload.get("sub")
            
        except JWTError:
            return None