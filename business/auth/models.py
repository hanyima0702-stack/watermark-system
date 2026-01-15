"""
认证相关数据模型
定义认证过程中使用的数据结构
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator
from enum import Enum


class AuthMethod(str, Enum):
    """认证方式枚举"""
    LOCAL = "local"
    LDAP = "ldap"
    SSO = "sso"


class MFAMethod(str, Enum):
    """多因素认证方式枚举"""
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    OPERATOR = "operator"
    AUDITOR = "auditor"
    USER = "user"


class Permission(str, Enum):
    """权限枚举"""
    # 文件操作权限
    FILE_UPLOAD = "file:upload"
    FILE_DOWNLOAD = "file:download"
    FILE_DELETE = "file:delete"
    FILE_VIEW = "file:view"
    
    # 水印操作权限
    WATERMARK_EMBED = "watermark:embed"
    WATERMARK_EXTRACT = "watermark:extract"
    WATERMARK_CONFIG = "watermark:config"
    
    # 系统管理权限
    USER_MANAGE = "user:manage"
    SYSTEM_CONFIG = "system:config"
    AUDIT_VIEW = "audit:view"
    
    # 报告权限
    REPORT_GENERATE = "report:generate"
    REPORT_VIEW = "report:view"


class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str
    password: str
    auth_method: AuthMethod = AuthMethod.LOCAL
    mfa_code: Optional[str] = None
    remember_me: bool = False
    
    @validator('username')
    def validate_username(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('用户名不能为空')
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 6:
            raise ValueError('密码长度不能少于6位')
        return v


class TokenResponse(BaseModel):
    """令牌响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    user_info: 'AuthUser'
    mfa_required: bool = False
    mfa_challenge: Optional['MFAChallenge'] = None


class AuthUser(BaseModel):
    """认证用户信息模型"""
    user_id: str
    username: str
    email: EmailStr
    department: Optional[str] = None
    roles: List[UserRole] = []
    permissions: List[Permission] = []
    is_active: bool = True
    last_login: Optional[datetime] = None
    auth_method: AuthMethod = AuthMethod.LOCAL
    mfa_enabled: bool = False
    
    def has_role(self, role: UserRole) -> bool:
        """检查用户是否具有指定角色"""
        return role in self.roles
    
    def has_permission(self, permission: Permission) -> bool:
        """检查用户是否具有指定权限"""
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        """检查是否为管理员"""
        return UserRole.ADMIN in self.roles
    
    def can_access_resource(self, resource: str, action: str) -> bool:
        """检查是否可以访问指定资源"""
        # 管理员拥有所有权限
        if self.is_admin():
            return True
        
        # 根据资源和操作构建权限字符串
        permission_str = f"{resource}:{action}"
        try:
            permission = Permission(permission_str)
            return self.has_permission(permission)
        except ValueError:
            return False


class MFAChallenge(BaseModel):
    """多因素认证挑战模型"""
    challenge_id: str
    method: MFAMethod
    expires_at: datetime
    qr_code: Optional[str] = None  # TOTP二维码
    backup_codes: Optional[List[str]] = None
    
    def is_expired(self) -> bool:
        """检查挑战是否已过期"""
        return datetime.utcnow() > self.expires_at


class MFASetupRequest(BaseModel):
    """MFA设置请求模型"""
    method: MFAMethod
    phone_number: Optional[str] = None
    email: Optional[EmailStr] = None
    
    @validator('phone_number')
    def validate_phone_number(cls, v, values):
        if values.get('method') == MFAMethod.SMS and not v:
            raise ValueError('SMS认证需要提供手机号码')
        return v
    
    @validator('email')
    def validate_email(cls, v, values):
        if values.get('method') == MFAMethod.EMAIL and not v:
            raise ValueError('邮件认证需要提供邮箱地址')
        return v


class PasswordChangeRequest(BaseModel):
    """密码修改请求模型"""
    current_password: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('新密码和确认密码不匹配')
        return v
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('新密码长度不能少于8位')
        
        # 检查密码复杂度
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v)
        
        if not (has_upper and has_lower and has_digit and has_special):
            raise ValueError('密码必须包含大写字母、小写字母、数字和特殊字符')
        
        return v


class RolePermissionMapping(BaseModel):
    """角色权限映射模型"""
    role: UserRole
    permissions: List[Permission]
    description: Optional[str] = None


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """检查会话是否已过期"""
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.utcnow() - self.last_activity > timeout


class AuditEvent(BaseModel):
    """审计事件模型"""
    event_id: str
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str] = None
    result: str  # success, failure, denied
    details: Dict[str, Any] = {}
    ip_address: str
    user_agent: str
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)


# 更新前向引用
TokenResponse.model_rebuild()
MFAChallenge.model_rebuild()