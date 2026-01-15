"""
LDAP客户端
提供LDAP/Active Directory集成功能
"""

import ldap3
from typing import Optional, Dict, List, Any
from ldap3 import Server, Connection, ALL, SUBTREE
from ldap3.core.exceptions import LDAPException

from ...shared.config import get_settings
from .models import AuthUser, UserRole, Permission


class LDAPClient:
    """LDAP客户端"""
    
    def __init__(self):
        self.settings = get_settings()
        self.ldap_config = self.settings.ldap
        self.server = None
        self.connection = None
        
        if self.ldap_config.enabled and self.ldap_config.server:
            self._initialize_connection()
    
    def _initialize_connection(self):
        """初始化LDAP连接"""
        try:
            self.server = Server(
                self.ldap_config.server,
                get_info=ALL,
                use_ssl=True if self.ldap_config.server.startswith('ldaps://') else False
            )
            
            # 创建绑定连接
            if self.ldap_config.bind_dn and self.ldap_config.bind_password:
                self.connection = Connection(
                    self.server,
                    user=self.ldap_config.bind_dn,
                    password=self.ldap_config.bind_password,
                    auto_bind=True
                )
            else:
                # 匿名连接
                self.connection = Connection(self.server, auto_bind=True)
                
        except LDAPException as e:
            print(f"LDAP连接初始化失败: {e}")
            self.server = None
            self.connection = None
    
    def is_enabled(self) -> bool:
        """检查LDAP是否启用"""
        return self.ldap_config.enabled and self.connection is not None
    
    def authenticate(self, username: str, password: str) -> Optional[AuthUser]:
        """LDAP身份认证"""
        if not self.is_enabled():
            return None
        
        try:
            # 搜索用户
            user_info = self._search_user(username)
            if not user_info:
                return None
            
            user_dn = user_info.get('dn')
            if not user_dn:
                return None
            
            # 尝试使用用户凭据绑定
            user_connection = Connection(
                self.server,
                user=user_dn,
                password=password,
                auto_bind=True
            )
            
            if user_connection.bind():
                # 认证成功，构建用户对象
                auth_user = self._build_auth_user(user_info)
                user_connection.unbind()
                return auth_user
            
        except LDAPException as e:
            print(f"LDAP认证失败: {e}")
        
        return None
    
    def _search_user(self, username: str) -> Optional[Dict[str, Any]]:
        """搜索用户信息"""
        if not self.connection:
            return None
        
        try:
            # 构建搜索过滤器
            search_filter = f"(|(sAMAccountName={username})(userPrincipalName={username})(cn={username}))"
            
            # 执行搜索
            success = self.connection.search(
                search_base=self.ldap_config.user_search_base or self.ldap_config.base_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=['cn', 'sAMAccountName', 'userPrincipalName', 'mail', 
                           'department', 'memberOf', 'displayName', 'telephoneNumber']
            )
            
            if success and self.connection.entries:
                entry = self.connection.entries[0]
                return {
                    'dn': entry.entry_dn,
                    'cn': str(entry.cn) if entry.cn else None,
                    'username': str(entry.sAMAccountName) if entry.sAMAccountName else username,
                    'email': str(entry.mail) if entry.mail else None,
                    'department': str(entry.department) if entry.department else None,
                    'display_name': str(entry.displayName) if entry.displayName else None,
                    'phone': str(entry.telephoneNumber) if entry.telephoneNumber else None,
                    'groups': [str(group) for group in entry.memberOf] if entry.memberOf else []
                }
        
        except LDAPException as e:
            print(f"LDAP用户搜索失败: {e}")
        
        return None
    
    def _build_auth_user(self, user_info: Dict[str, Any]) -> AuthUser:
        """构建认证用户对象"""
        # 从LDAP组映射到系统角色
        roles = self._map_groups_to_roles(user_info.get('groups', []))
        
        # 根据角色分配权限
        permissions = self._get_permissions_for_roles(roles)
        
        return AuthUser(
            user_id=user_info['username'],
            username=user_info['username'],
            email=user_info.get('email', f"{user_info['username']}@company.com"),
            department=user_info.get('department'),
            roles=roles,
            permissions=permissions,
            is_active=True,
            auth_method='ldap'
        )
    
    def _map_groups_to_roles(self, groups: List[str]) -> List[UserRole]:
        """将LDAP组映射到系统角色"""
        roles = []
        
        # 定义组到角色的映射关系
        group_role_mapping = {
            'CN=Watermark_Admins': UserRole.ADMIN,
            'CN=Watermark_Operators': UserRole.OPERATOR,
            'CN=Watermark_Auditors': UserRole.AUDITOR,
            'CN=Watermark_Users': UserRole.USER
        }
        
        for group in groups:
            # 提取组的CN部分
            group_cn = self._extract_cn_from_dn(group)
            if group_cn in group_role_mapping:
                role = group_role_mapping[group_cn]
                if role not in roles:
                    roles.append(role)
        
        # 如果没有匹配到任何角色，默认为普通用户
        if not roles:
            roles.append(UserRole.USER)
        
        return roles
    
    def _extract_cn_from_dn(self, dn: str) -> str:
        """从DN中提取CN"""
        try:
            # 解析DN，提取第一个CN
            parts = dn.split(',')
            for part in parts:
                part = part.strip()
                if part.startswith('CN='):
                    return part
        except:
            pass
        return dn
    
    def _get_permissions_for_roles(self, roles: List[UserRole]) -> List[Permission]:
        """根据角色获取权限"""
        permissions = set()
        
        # 角色权限映射
        role_permissions = {
            UserRole.ADMIN: [
                Permission.FILE_UPLOAD, Permission.FILE_DOWNLOAD, Permission.FILE_DELETE, Permission.FILE_VIEW,
                Permission.WATERMARK_EMBED, Permission.WATERMARK_EXTRACT, Permission.WATERMARK_CONFIG,
                Permission.USER_MANAGE, Permission.SYSTEM_CONFIG, Permission.AUDIT_VIEW,
                Permission.REPORT_GENERATE, Permission.REPORT_VIEW
            ],
            UserRole.OPERATOR: [
                Permission.FILE_UPLOAD, Permission.FILE_DOWNLOAD, Permission.FILE_VIEW,
                Permission.WATERMARK_EMBED, Permission.WATERMARK_EXTRACT, Permission.WATERMARK_CONFIG,
                Permission.REPORT_GENERATE, Permission.REPORT_VIEW
            ],
            UserRole.AUDITOR: [
                Permission.FILE_VIEW, Permission.WATERMARK_EXTRACT,
                Permission.AUDIT_VIEW, Permission.REPORT_VIEW
            ],
            UserRole.USER: [
                Permission.FILE_UPLOAD, Permission.FILE_DOWNLOAD, Permission.FILE_VIEW,
                Permission.WATERMARK_EMBED, Permission.REPORT_VIEW
            ]
        }
        
        for role in roles:
            if role in role_permissions:
                permissions.update(role_permissions[role])
        
        return list(permissions)
    
    def get_user_groups(self, username: str) -> List[str]:
        """获取用户所属组"""
        user_info = self._search_user(username)
        if user_info:
            return user_info.get('groups', [])
        return []
    
    def search_users(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """搜索用户"""
        if not self.connection:
            return []
        
        try:
            # 构建搜索过滤器
            search_filter = f"(|(cn=*{search_term}*)(sAMAccountName=*{search_term}*)(mail=*{search_term}*))"
            
            success = self.connection.search(
                search_base=self.ldap_config.user_search_base or self.ldap_config.base_dn,
                search_filter=search_filter,
                search_scope=SUBTREE,
                attributes=['cn', 'sAMAccountName', 'mail', 'department', 'displayName'],
                size_limit=limit
            )
            
            if success:
                users = []
                for entry in self.connection.entries:
                    users.append({
                        'username': str(entry.sAMAccountName) if entry.sAMAccountName else None,
                        'display_name': str(entry.displayName) if entry.displayName else None,
                        'email': str(entry.mail) if entry.mail else None,
                        'department': str(entry.department) if entry.department else None
                    })
                return users
        
        except LDAPException as e:
            print(f"LDAP用户搜索失败: {e}")
        
        return []
    
    def validate_group_membership(self, username: str, required_group: str) -> bool:
        """验证用户是否属于指定组"""
        user_groups = self.get_user_groups(username)
        return any(required_group in group for group in user_groups)
    
    def close_connection(self):
        """关闭LDAP连接"""
        if self.connection:
            self.connection.unbind()
            self.connection = None