"""
身份认证服务
统一的身份认证入口，支持本地认证、LDAP认证和MFA
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ...shared.interfaces import IAuthService, UserInfo
from ...storage.dao.user_dao import UserDAO
from .models import (
    AuthUser, LoginRequest, TokenResponse, MFAChallenge, 
    AuthMethod, UserRole, Permission, SessionInfo, AuditEvent
)
from .jwt_manager import JWTManager
from .ldap_client import LDAPClient
from .mfa_service import MFAService


class AuthService(IAuthService):
    """身份认证服务"""
    
    def __init__(self):
        self.user_dao = UserDAO()
        self.jwt_manager = JWTManager()
        self.ldap_client = LDAPClient()
        self.mfa_service = MFAService()
        
        # 会话管理（实际应该使用Redis）
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_accounts: Dict[str, datetime] = {}
    
    async def login(self, request: LoginRequest, ip_address: str, user_agent: str) -> TokenResponse:
        """用户登录"""
        # 检查账户是否被锁定
        if self._is_account_locked(request.username):
            raise Exception("账户已被锁定，请稍后再试")
        
        try:
            # 根据认证方式进行身份验证
            if request.auth_method == AuthMethod.LDAP:
                user = await self._authenticate_ldap(request.username, request.password)
            else:
                user = await self._authenticate_local(request.username, request.password)
            
            if not user:
                self._record_failed_attempt(request.username)
                raise Exception("用户名或密码错误")
            
            # 检查用户是否激活
            if not user.is_active:
                raise Exception("账户已被禁用")
            
            # 清除失败尝试记录
            self._clear_failed_attempts(request.username)
            
            # 检查是否需要MFA
            if user.mfa_enabled and not request.mfa_code:
                mfa_challenge = await self._initiate_mfa_challenge(user.user_id)
                return TokenResponse(
                    access_token="",
                    expires_in=0,
                    user_info=user,
                    mfa_required=True,
                    mfa_challenge=mfa_challenge
                )
            
            # 验证MFA（如果提供了代码）
            if user.mfa_enabled and request.mfa_code:
                if not await self._verify_mfa(user.user_id, request.mfa_code):
                    raise Exception("MFA验证失败")
            
            # 生成令牌
            access_token = self.jwt_manager.create_access_token(user)
            refresh_token = None
            
            if request.remember_me:
                refresh_token = self.jwt_manager.create_refresh_token(user.user_id)
            
            # 创建会话
            session_id = await self._create_session(user.user_id, ip_address, user_agent)
            
            # 更新最后登录时间
            await self._update_last_login(user.user_id)
            
            # 记录审计日志
            await self._log_audit_event(
                user.user_id, "login", "session", session_id,
                "success", {"auth_method": request.auth_method.value},
                ip_address, user_agent
            )
            
            return TokenResponse(
                access_token=access_token,
                expires_in=self.jwt_manager.access_token_expire_minutes * 60,
                refresh_token=refresh_token,
                user_info=user
            )
            
        except Exception as e:
            # 记录失败的审计日志
            await self._log_audit_event(
                request.username, "login", "session", None,
                "failure", {"error": str(e), "auth_method": request.auth_method.value},
                ip_address, user_agent
            )
            raise
    
    async def logout(self, token: str, ip_address: str, user_agent: str) -> bool:
        """用户登出"""
        try:
            # 验证令牌
            payload = self.jwt_manager.verify_token(token)
            if not payload:
                return False
            
            user_id = payload.get("sub")
            
            # 撤销令牌
            self.jwt_manager.revoke_token(token)
            
            # 删除会话
            await self._remove_session(user_id)
            
            # 记录审计日志
            await self._log_audit_event(
                user_id, "logout", "session", None,
                "success", {}, ip_address, user_agent
            )
            
            return True
            
        except Exception:
            return False
    
    async def authenticate(self, token: str) -> Optional[UserInfo]:
        """验证用户身份"""
        try:
            # 验证令牌
            payload = self.jwt_manager.verify_token(token)
            if not payload:
                return None
            
            # 检查令牌是否在黑名单中
            if self.jwt_manager.is_token_blacklisted(token):
                return None
            
            user_id = payload.get("sub")
            
            # 检查会话是否有效
            if not await self._is_session_valid(user_id):
                return None
            
            # 构建用户信息
            return UserInfo(
                user_id=user_id,
                username=payload.get("username"),
                department=payload.get("department"),
                email=payload.get("email"),
                roles=payload.get("roles", []),
                created_at=datetime.utcnow()
            )
            
        except Exception:
            return None
    
    async def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """检查用户权限"""
        try:
            # 获取用户信息
            user = await self.get_user_info(user_id)
            if not user:
                return False
            
            # 管理员拥有所有权限
            if "admin" in user.roles:
                return True
            
            # 检查具体权限
            permission_str = f"{resource}:{action}"
            
            # 这里应该从数据库或缓存中获取用户的详细权限
            # 暂时使用简化的权限检查逻辑
            return await self._check_permission(user_id, permission_str)
            
        except Exception:
            return False
    
    async def get_user_info(self, user_id: str) -> Optional[UserInfo]:
        """获取用户信息"""
        try:
            user = await self.user_dao.get_by_id(user_id)
            if not user:
                return None
            
            return UserInfo(
                user_id=user.user_id,
                username=user.username,
                department=user.department,
                email=user.email,
                roles=user.roles or [],
                created_at=user.created_at
            )
            
        except Exception:
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[TokenResponse]:
        """刷新访问令牌"""
        try:
            user_id = self.jwt_manager.verify_refresh_token(refresh_token)
            if not user_id:
                return None
            
            # 获取用户信息
            user_info = await self.get_user_info(user_id)
            if not user_info:
                return None
            
            # 构建AuthUser对象
            user = AuthUser(
                user_id=user_info.user_id,
                username=user_info.username,
                email=user_info.email,
                department=user_info.department,
                roles=[UserRole(role) for role in user_info.roles],
                permissions=[]  # 这里应该从数据库获取权限
            )
            
            return self.jwt_manager.refresh_access_token(refresh_token, user)
            
        except Exception:
            return None
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """修改密码"""
        try:
            # 获取用户
            user = await self.user_dao.get_by_id(user_id)
            if not user:
                return False
            
            # 验证当前密码
            if not self.jwt_manager.verify_password(current_password, user.password_hash):
                return False
            
            # 更新密码
            new_password_hash = self.jwt_manager.hash_password(new_password)
            await self.user_dao.update_password(user_id, new_password_hash)
            
            # 记录审计日志
            await self._log_audit_event(
                user_id, "password_change", "user", user_id,
                "success", {}, "", ""
            )
            
            return True
            
        except Exception:
            return False
    
    async def setup_mfa(self, user_id: str, method: str, **kwargs) -> MFAChallenge:
        """设置MFA"""
        from .models import MFAMethod
        
        mfa_method = MFAMethod(method)
        
        if mfa_method == MFAMethod.TOTP:
            user = await self.user_dao.get_by_id(user_id)
            return self.mfa_service.setup_totp(user_id, user.username)
        elif mfa_method == MFAMethod.SMS:
            phone = kwargs.get('phone_number')
            return self.mfa_service.setup_sms(user_id, phone)
        elif mfa_method == MFAMethod.EMAIL:
            email = kwargs.get('email')
            return self.mfa_service.setup_email(user_id, email)
        else:
            raise Exception(f"不支持的MFA方法: {method}")
    
    # 私有方法
    async def _authenticate_local(self, username: str, password: str) -> Optional[AuthUser]:
        """本地认证"""
        try:
            user = await self.user_dao.get_by_username(username)
            if not user:
                return None
            
            if not self.jwt_manager.verify_password(password, user.password_hash):
                return None
            
            return AuthUser(
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                department=user.department,
                roles=[UserRole(role) for role in (user.roles or [])],
                permissions=await self._get_user_permissions(user.user_id),
                is_active=user.is_active,
                auth_method=AuthMethod.LOCAL,
                mfa_enabled=self.mfa_service.is_mfa_enabled(user.user_id)
            )
            
        except Exception:
            return None
    
    async def _authenticate_ldap(self, username: str, password: str) -> Optional[AuthUser]:
        """LDAP认证"""
        if not self.ldap_client.is_enabled():
            return None
        
        return self.ldap_client.authenticate(username, password)
    
    async def _initiate_mfa_challenge(self, user_id: str) -> MFAChallenge:
        """发起MFA挑战"""
        # 获取用户启用的MFA方法
        methods = self.mfa_service.get_user_mfa_methods(user_id)
        if not methods:
            raise Exception("用户未启用MFA")
        
        # 使用第一个可用的方法
        method = methods[0]
        return self.mfa_service.generate_mfa_challenge(user_id, method)
    
    async def _verify_mfa(self, user_id: str, code: str, challenge_id: str = None) -> bool:
        """验证MFA"""
        if challenge_id:
            return self.mfa_service.verify_mfa_code(user_id, challenge_id, code)
        else:
            # 对于TOTP，不需要challenge_id
            return self.mfa_service.verify_totp(user_id, code)
    
    async def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        session = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    async def _remove_session(self, user_id: str):
        """删除会话"""
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
    
    async def _is_session_valid(self, user_id: str) -> bool:
        """检查会话是否有效"""
        for session in self.active_sessions.values():
            if session.user_id == user_id and not session.is_expired():
                # 更新最后活动时间
                session.last_activity = datetime.utcnow()
                return True
        return False
    
    async def _update_last_login(self, user_id: str):
        """更新最后登录时间"""
        await self.user_dao.update_last_login(user_id, datetime.utcnow())
    
    async def _get_user_permissions(self, user_id: str) -> list:
        """获取用户权限"""
        # 这里应该从数据库获取用户的详细权限
        # 暂时返回空列表
        return []
    
    async def _check_permission(self, user_id: str, permission: str) -> bool:
        """检查权限"""
        # 这里应该实现具体的权限检查逻辑
        return True
    
    def _is_account_locked(self, username: str) -> bool:
        """检查账户是否被锁定"""
        if username in self.locked_accounts:
            lock_time = self.locked_accounts[username]
            if datetime.utcnow() - lock_time < timedelta(minutes=30):
                return True
            else:
                # 锁定时间已过，移除锁定
                del self.locked_accounts[username]
        return False
    
    def _record_failed_attempt(self, username: str):
        """记录失败尝试"""
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        
        # 如果失败次数达到阈值，锁定账户
        if self.failed_attempts[username] >= 5:
            self.locked_accounts[username] = datetime.utcnow()
    
    def _clear_failed_attempts(self, username: str):
        """清除失败尝试记录"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    async def _log_audit_event(
        self, 
        user_id: str, 
        action: str, 
        resource: str, 
        resource_id: Optional[str],
        result: str, 
        details: Dict[str, Any],
        ip_address: str, 
        user_agent: str
    ):
        """记录审计事件"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            result=result,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # 这里应该将审计事件存储到数据库或日志系统
        print(f"审计事件: {event}")  # 临时打印，实际应该存储