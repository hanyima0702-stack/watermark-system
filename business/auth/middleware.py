"""
认证和权限控制中间件
提供FastAPI中间件和装饰器，用于身份认证和权限检查
"""

import functools
from typing import Optional, List, Callable, Any
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...shared.interfaces import UserInfo
from .auth_service import AuthService
from .rbac_manager import RBACManager
from .models import UserRole, Permission


# 全局认证服务实例
auth_service = AuthService()
rbac_manager = RBACManager()

# HTTP Bearer认证方案
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """身份认证中间件"""
    
    def __init__(self, app, exclude_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs", "/redoc", "/openapi.json", 
            "/health", "/metrics", "/api/v1/auth/login"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        # 检查是否为排除路径
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # 获取Authorization头
        authorization = request.headers.get("Authorization")
        if not authorization:
            return self._unauthorized_response("缺少Authorization头")
        
        # 提取token
        try:
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                return self._unauthorized_response("无效的认证方案")
        except ValueError:
            return self._unauthorized_response("无效的Authorization头格式")
        
        # 验证token
        user_info = await auth_service.authenticate(token)
        if not user_info:
            return self._unauthorized_response("无效的访问令牌")
        
        # 将用户信息添加到请求状态
        request.state.user = user_info
        request.state.token = token
        
        return await call_next(request)
    
    def _unauthorized_response(self, message: str) -> Response:
        """返回未授权响应"""
        return Response(
            content=f'{{"error": "{message}"}}',
            status_code=401,
            headers={"Content-Type": "application/json"}
        )


class RBACMiddleware(BaseHTTPMiddleware):
    """基于角色的访问控制中间件"""
    
    def __init__(self, app, resource_mapping: Optional[dict] = None):
        super().__init__(app)
        # 路径到资源的映射
        self.resource_mapping = resource_mapping or {
            "/api/v1/files": "file",
            "/api/v1/watermark": "watermark",
            "/api/v1/users": "user",
            "/api/v1/system": "system",
            "/api/v1/audit": "audit",
            "/api/v1/reports": "report",
            "/api/v1/config": "config"
        }
        
        # HTTP方法到操作的映射
        self.method_mapping = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete"
        }
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        # 检查是否已通过身份认证
        if not hasattr(request.state, "user"):
            return await call_next(request)
        
        user_info = request.state.user
        
        # 确定资源和操作
        resource = self._get_resource_from_path(request.url.path)
        action = self.method_mapping.get(request.method, "read")
        
        if resource:
            # 构建权限检查上下文
            context = {
                "current_user": user_info.user_id,
                "request_path": request.url.path,
                "request_method": request.method
            }
            
            # 检查权限
            user_roles = [UserRole(role) for role in user_info.roles]
            has_permission = rbac_manager.check_permission(
                user_roles, resource, action, context
            )
            
            if not has_permission:
                return Response(
                    content=f'{{"error": "权限不足"}}',
                    status_code=403,
                    headers={"Content-Type": "application/json"}
                )
        
        return await call_next(request)
    
    def _get_resource_from_path(self, path: str) -> Optional[str]:
        """从路径获取资源类型"""
        for path_prefix, resource in self.resource_mapping.items():
            if path.startswith(path_prefix):
                return resource
        return None


# 依赖注入函数
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> UserInfo:
    """获取当前用户（依赖注入）"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_info = await auth_service.authenticate(credentials.credentials)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_info


async def get_current_active_user(
    current_user: UserInfo = Depends(get_current_user)
) -> UserInfo:
    """获取当前活跃用户"""
    # 这里可以添加额外的用户状态检查
    return current_user


# 权限检查装饰器
def require_permission(resource: str, action: str):
    """权限检查装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 从kwargs中获取current_user
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="未认证的用户"
                )
            
            # 检查权限
            user_roles = [UserRole(role) for role in current_user.roles]
            has_permission = rbac_manager.check_permission(
                user_roles, resource, action
            )
            
            if not has_permission:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足: 需要 {resource}:{action} 权限"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(required_roles: List[UserRole]):
    """角色检查装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="未认证的用户"
                )
            
            user_roles = [UserRole(role) for role in current_user.roles]
            has_required_role = any(role in user_roles for role in required_roles)
            
            if not has_required_role:
                role_names = [role.value for role in required_roles]
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"权限不足: 需要以下角色之一 {role_names}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def admin_required(func: Callable) -> Callable:
    """管理员权限装饰器"""
    return require_role([UserRole.ADMIN])(func)


def operator_or_admin_required(func: Callable) -> Callable:
    """操作员或管理员权限装饰器"""
    return require_role([UserRole.OPERATOR, UserRole.ADMIN])(func)


# 权限检查依赖
class PermissionChecker:
    """权限检查器"""
    
    def __init__(self, resource: str, action: str):
        self.resource = resource
        self.action = action
    
    def __call__(self, current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
        """检查权限并返回用户信息"""
        user_roles = [UserRole(role) for role in current_user.roles]
        has_permission = rbac_manager.check_permission(
            user_roles, self.resource, self.action
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足: 需要 {self.resource}:{self.action} 权限"
            )
        
        return current_user


# 常用权限检查器实例
require_file_read = PermissionChecker("file", "read")
require_file_create = PermissionChecker("file", "create")
require_file_update = PermissionChecker("file", "update")
require_file_delete = PermissionChecker("file", "delete")

require_watermark_embed = PermissionChecker("watermark", "create")
require_watermark_extract = PermissionChecker("watermark", "execute")
require_watermark_config = PermissionChecker("watermark", "update")

require_user_manage = PermissionChecker("user", "manage")
require_system_config = PermissionChecker("system", "update")
require_audit_view = PermissionChecker("audit", "read")

require_report_generate = PermissionChecker("report", "create")
require_report_view = PermissionChecker("report", "read")


# 资源所有权检查
class ResourceOwnershipChecker:
    """资源所有权检查器"""
    
    def __init__(self, resource: str, action: str, owner_field: str = "owner_id"):
        self.resource = resource
        self.action = action
        self.owner_field = owner_field
    
    def __call__(
        self, 
        resource_id: str,
        current_user: UserInfo = Depends(get_current_user)
    ) -> UserInfo:
        """检查资源所有权"""
        # 管理员跳过所有权检查
        if UserRole.ADMIN.value in current_user.roles:
            return current_user
        
        # 构建权限检查上下文
        context = {
            "current_user": current_user.user_id,
            "resource_owner": self._get_resource_owner(resource_id),
            "owner": True  # 表示需要检查所有权
        }
        
        user_roles = [UserRole(role) for role in current_user.roles]
        has_permission = rbac_manager.check_permission(
            user_roles, self.resource, self.action, context
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足: 只能访问自己的资源"
            )
        
        return current_user
    
    def _get_resource_owner(self, resource_id: str) -> Optional[str]:
        """获取资源所有者（需要根据具体资源类型实现）"""
        # 这里应该根据resource_id查询数据库获取资源所有者
        # 暂时返回None，实际使用时需要实现
        return None


# 批量权限检查
async def check_bulk_permissions(
    user_roles: List[UserRole],
    permissions: List[tuple]  # [(resource, action), ...]
) -> dict:
    """批量检查权限"""
    results = {}
    
    for resource, action in permissions:
        permission_key = f"{resource}:{action}"
        results[permission_key] = rbac_manager.check_permission(
            user_roles, resource, action
        )
    
    return results


# 权限信息获取
async def get_user_permissions_info(
    current_user: UserInfo = Depends(get_current_user)
) -> dict:
    """获取用户权限信息"""
    user_roles = [UserRole(role) for role in current_user.roles]
    permissions = rbac_manager.get_user_permissions(user_roles)
    
    return {
        "user_id": current_user.user_id,
        "roles": current_user.roles,
        "permissions": permissions,
        "is_admin": UserRole.ADMIN.value in current_user.roles
    }