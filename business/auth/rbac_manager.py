"""
基于角色的访问控制(RBAC)管理器
实现角色权限模型和权限检查机制
"""

from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from .models import UserRole, Permission, RolePermissionMapping


class ResourceType(str, Enum):
    """资源类型枚举"""
    FILE = "file"
    WATERMARK = "watermark"
    USER = "user"
    SYSTEM = "system"
    AUDIT = "audit"
    REPORT = "report"
    CONFIG = "config"


class ActionType(str, Enum):
    """操作类型枚举"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"


@dataclass
class PermissionRule:
    """权限规则"""
    resource: ResourceType
    action: ActionType
    conditions: Optional[Dict[str, any]] = None
    description: str = ""


@dataclass
class RoleDefinition:
    """角色定义"""
    role: UserRole
    name: str
    description: str
    permissions: List[PermissionRule]
    inherits_from: Optional[UserRole] = None


class RBACManager:
    """RBAC管理器"""
    
    def __init__(self):
        self.role_definitions = self._initialize_role_definitions()
        self.role_hierarchy = self._build_role_hierarchy()
        self.permission_cache: Dict[str, Set[str]] = {}
    
    def _initialize_role_definitions(self) -> Dict[UserRole, RoleDefinition]:
        """初始化角色定义"""
        roles = {}
        
        # 普通用户角色
        roles[UserRole.USER] = RoleDefinition(
            role=UserRole.USER,
            name="普通用户",
            description="可以上传文件、嵌入水印、查看自己的报告",
            permissions=[
                PermissionRule(ResourceType.FILE, ActionType.CREATE, description="上传文件"),
                PermissionRule(ResourceType.FILE, ActionType.READ, {"owner": True}, "查看自己的文件"),
                PermissionRule(ResourceType.FILE, ActionType.UPDATE, {"owner": True}, "更新自己的文件"),
                PermissionRule(ResourceType.WATERMARK, ActionType.CREATE, description="嵌入水印"),
                PermissionRule(ResourceType.WATERMARK, ActionType.READ, {"owner": True}, "查看自己的水印配置"),
                PermissionRule(ResourceType.REPORT, ActionType.READ, {"owner": True}, "查看自己的报告"),
            ]
        )
        
        # 操作员角色
        roles[UserRole.OPERATOR] = RoleDefinition(
            role=UserRole.OPERATOR,
            name="操作员",
            description="可以处理水印任务、管理水印配置、生成报告",
            permissions=[
                PermissionRule(ResourceType.FILE, ActionType.CREATE, description="上传文件"),
                PermissionRule(ResourceType.FILE, ActionType.READ, description="查看所有文件"),
                PermissionRule(ResourceType.FILE, ActionType.UPDATE, description="更新文件"),
                PermissionRule(ResourceType.FILE, ActionType.DELETE, {"owner": True}, "删除自己的文件"),
                PermissionRule(ResourceType.WATERMARK, ActionType.CREATE, description="嵌入水印"),
                PermissionRule(ResourceType.WATERMARK, ActionType.READ, description="查看水印配置"),
                PermissionRule(ResourceType.WATERMARK, ActionType.UPDATE, description="更新水印配置"),
                PermissionRule(ResourceType.WATERMARK, ActionType.EXECUTE, description="提取水印"),
                PermissionRule(ResourceType.CONFIG, ActionType.READ, description="查看配置"),
                PermissionRule(ResourceType.CONFIG, ActionType.UPDATE, {"scope": "watermark"}, "更新水印配置"),
                PermissionRule(ResourceType.REPORT, ActionType.CREATE, description="生成报告"),
                PermissionRule(ResourceType.REPORT, ActionType.READ, description="查看报告"),
            ],
            inherits_from=UserRole.USER
        )
        
        # 审计员角色
        roles[UserRole.AUDITOR] = RoleDefinition(
            role=UserRole.AUDITOR,
            name="审计员",
            description="可以查看审计日志、提取水印、查看所有报告",
            permissions=[
                PermissionRule(ResourceType.FILE, ActionType.READ, description="查看文件"),
                PermissionRule(ResourceType.WATERMARK, ActionType.EXECUTE, description="提取水印"),
                PermissionRule(ResourceType.AUDIT, ActionType.READ, description="查看审计日志"),
                PermissionRule(ResourceType.REPORT, ActionType.READ, description="查看所有报告"),
                PermissionRule(ResourceType.REPORT, ActionType.CREATE, {"type": "audit"}, "生成审计报告"),
            ]
        )
        
        # 管理员角色
        roles[UserRole.ADMIN] = RoleDefinition(
            role=UserRole.ADMIN,
            name="管理员",
            description="拥有系统的完全访问权限",
            permissions=[
                PermissionRule(ResourceType.FILE, ActionType.CREATE, description="上传文件"),
                PermissionRule(ResourceType.FILE, ActionType.READ, description="查看所有文件"),
                PermissionRule(ResourceType.FILE, ActionType.UPDATE, description="更新文件"),
                PermissionRule(ResourceType.FILE, ActionType.DELETE, description="删除文件"),
                PermissionRule(ResourceType.WATERMARK, ActionType.CREATE, description="嵌入水印"),
                PermissionRule(ResourceType.WATERMARK, ActionType.READ, description="查看水印配置"),
                PermissionRule(ResourceType.WATERMARK, ActionType.UPDATE, description="更新水印配置"),
                PermissionRule(ResourceType.WATERMARK, ActionType.DELETE, description="删除水印配置"),
                PermissionRule(ResourceType.WATERMARK, ActionType.EXECUTE, description="提取水印"),
                PermissionRule(ResourceType.USER, ActionType.CREATE, description="创建用户"),
                PermissionRule(ResourceType.USER, ActionType.READ, description="查看用户"),
                PermissionRule(ResourceType.USER, ActionType.UPDATE, description="更新用户"),
                PermissionRule(ResourceType.USER, ActionType.DELETE, description="删除用户"),
                PermissionRule(ResourceType.USER, ActionType.MANAGE, description="管理用户"),
                PermissionRule(ResourceType.SYSTEM, ActionType.READ, description="查看系统配置"),
                PermissionRule(ResourceType.SYSTEM, ActionType.UPDATE, description="更新系统配置"),
                PermissionRule(ResourceType.SYSTEM, ActionType.MANAGE, description="管理系统"),
                PermissionRule(ResourceType.AUDIT, ActionType.READ, description="查看审计日志"),
                PermissionRule(ResourceType.REPORT, ActionType.CREATE, description="生成报告"),
                PermissionRule(ResourceType.REPORT, ActionType.READ, description="查看报告"),
                PermissionRule(ResourceType.REPORT, ActionType.UPDATE, description="更新报告"),
                PermissionRule(ResourceType.REPORT, ActionType.DELETE, description="删除报告"),
                PermissionRule(ResourceType.CONFIG, ActionType.READ, description="查看配置"),
                PermissionRule(ResourceType.CONFIG, ActionType.UPDATE, description="更新配置"),
                PermissionRule(ResourceType.CONFIG, ActionType.DELETE, description="删除配置"),
            ]
        )
        
        return roles
    
    def _build_role_hierarchy(self) -> Dict[UserRole, Set[UserRole]]:
        """构建角色继承层次结构"""
        hierarchy = {}
        
        for role, definition in self.role_definitions.items():
            inherited_roles = set()
            
            # 递归获取所有继承的角色
            def collect_inherited_roles(current_role: UserRole):
                role_def = self.role_definitions.get(current_role)
                if role_def and role_def.inherits_from:
                    inherited_roles.add(role_def.inherits_from)
                    collect_inherited_roles(role_def.inherits_from)
            
            collect_inherited_roles(role)
            hierarchy[role] = inherited_roles
        
        return hierarchy
    
    def check_permission(
        self, 
        user_roles: List[UserRole], 
        resource: str, 
        action: str,
        context: Optional[Dict[str, any]] = None
    ) -> bool:
        """检查用户是否有权限执行指定操作"""
        
        # 管理员拥有所有权限
        if UserRole.ADMIN in user_roles:
            return True
        
        # 获取所有有效角色（包括继承的角色）
        effective_roles = self._get_effective_roles(user_roles)
        
        # 检查每个角色的权限
        for role in effective_roles:
            if self._check_role_permission(role, resource, action, context):
                return True
        
        return False
    
    def _get_effective_roles(self, user_roles: List[UserRole]) -> Set[UserRole]:
        """获取用户的有效角色（包括继承的角色）"""
        effective_roles = set(user_roles)
        
        for role in user_roles:
            if role in self.role_hierarchy:
                effective_roles.update(self.role_hierarchy[role])
        
        return effective_roles
    
    def _check_role_permission(
        self, 
        role: UserRole, 
        resource: str, 
        action: str,
        context: Optional[Dict[str, any]] = None
    ) -> bool:
        """检查角色是否有指定权限"""
        role_def = self.role_definitions.get(role)
        if not role_def:
            return False
        
        try:
            resource_type = ResourceType(resource)
            action_type = ActionType(action)
        except ValueError:
            return False
        
        # 检查角色的权限规则
        for permission_rule in role_def.permissions:
            if (permission_rule.resource == resource_type and 
                permission_rule.action == action_type):
                
                # 检查条件
                if self._check_permission_conditions(permission_rule.conditions, context):
                    return True
        
        return False
    
    def _check_permission_conditions(
        self, 
        conditions: Optional[Dict[str, any]], 
        context: Optional[Dict[str, any]]
    ) -> bool:
        """检查权限条件"""
        if not conditions:
            return True
        
        if not context:
            return False
        
        for key, expected_value in conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            # 特殊处理owner条件
            if key == "owner" and expected_value is True:
                # 检查资源是否属于当前用户
                resource_owner = context.get("resource_owner")
                current_user = context.get("current_user")
                if resource_owner != current_user:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def get_user_permissions(self, user_roles: List[UserRole]) -> List[str]:
        """获取用户的所有权限"""
        permissions = set()
        effective_roles = self._get_effective_roles(user_roles)
        
        for role in effective_roles:
            role_def = self.role_definitions.get(role)
            if role_def:
                for permission_rule in role_def.permissions:
                    permission_str = f"{permission_rule.resource.value}:{permission_rule.action.value}"
                    permissions.add(permission_str)
        
        return list(permissions)
    
    def get_role_permissions(self, role: UserRole) -> List[PermissionRule]:
        """获取角色的权限"""
        role_def = self.role_definitions.get(role)
        if role_def:
            return role_def.permissions
        return []
    
    def get_all_roles(self) -> List[RoleDefinition]:
        """获取所有角色定义"""
        return list(self.role_definitions.values())
    
    def add_role_permission(self, role: UserRole, permission_rule: PermissionRule) -> bool:
        """为角色添加权限"""
        role_def = self.role_definitions.get(role)
        if role_def:
            # 检查权限是否已存在
            for existing_rule in role_def.permissions:
                if (existing_rule.resource == permission_rule.resource and
                    existing_rule.action == permission_rule.action and
                    existing_rule.conditions == permission_rule.conditions):
                    return False  # 权限已存在
            
            role_def.permissions.append(permission_rule)
            self._clear_permission_cache()
            return True
        return False
    
    def remove_role_permission(
        self, 
        role: UserRole, 
        resource: ResourceType, 
        action: ActionType,
        conditions: Optional[Dict[str, any]] = None
    ) -> bool:
        """移除角色权限"""
        role_def = self.role_definitions.get(role)
        if role_def:
            for i, permission_rule in enumerate(role_def.permissions):
                if (permission_rule.resource == resource and
                    permission_rule.action == action and
                    permission_rule.conditions == conditions):
                    del role_def.permissions[i]
                    self._clear_permission_cache()
                    return True
        return False
    
    def create_custom_role(
        self, 
        role_name: str, 
        description: str, 
        permissions: List[PermissionRule],
        inherits_from: Optional[UserRole] = None
    ) -> UserRole:
        """创建自定义角色"""
        # 这里简化处理，实际应该支持动态角色创建
        # 可以扩展UserRole枚举或使用字符串类型的角色
        raise NotImplementedError("自定义角色创建功能待实现")
    
    def validate_permission_string(self, permission: str) -> bool:
        """验证权限字符串格式"""
        try:
            parts = permission.split(":")
            if len(parts) != 2:
                return False
            
            resource, action = parts
            ResourceType(resource)
            ActionType(action)
            return True
        except ValueError:
            return False
    
    def get_permission_description(self, permission: str) -> Optional[str]:
        """获取权限描述"""
        try:
            parts = permission.split(":")
            if len(parts) != 2:
                return None
            
            resource, action = parts
            resource_type = ResourceType(resource)
            action_type = ActionType(action)
            
            # 查找匹配的权限规则
            for role_def in self.role_definitions.values():
                for permission_rule in role_def.permissions:
                    if (permission_rule.resource == resource_type and
                        permission_rule.action == action_type):
                        return permission_rule.description
            
            return f"{action} {resource}"
        except ValueError:
            return None
    
    def check_role_hierarchy(self, parent_role: UserRole, child_role: UserRole) -> bool:
        """检查角色继承关系"""
        child_inherited_roles = self.role_hierarchy.get(child_role, set())
        return parent_role in child_inherited_roles
    
    def get_role_hierarchy_tree(self) -> Dict[str, any]:
        """获取角色继承树"""
        tree = {}
        
        for role, definition in self.role_definitions.items():
            tree[role.value] = {
                "name": definition.name,
                "description": definition.description,
                "inherits_from": definition.inherits_from.value if definition.inherits_from else None,
                "permissions_count": len(definition.permissions)
            }
        
        return tree
    
    def _clear_permission_cache(self):
        """清除权限缓存"""
        self.permission_cache.clear()
    
    def export_role_definitions(self) -> Dict[str, any]:
        """导出角色定义（用于配置管理）"""
        export_data = {}
        
        for role, definition in self.role_definitions.items():
            permissions = []
            for perm_rule in definition.permissions:
                permissions.append({
                    "resource": perm_rule.resource.value,
                    "action": perm_rule.action.value,
                    "conditions": perm_rule.conditions,
                    "description": perm_rule.description
                })
            
            export_data[role.value] = {
                "name": definition.name,
                "description": definition.description,
                "inherits_from": definition.inherits_from.value if definition.inherits_from else None,
                "permissions": permissions
            }
        
        return export_data
    
    def import_role_definitions(self, import_data: Dict[str, any]) -> bool:
        """导入角色定义"""
        try:
            new_definitions = {}
            
            for role_str, role_data in import_data.items():
                role = UserRole(role_str)
                
                permissions = []
                for perm_data in role_data.get("permissions", []):
                    permission_rule = PermissionRule(
                        resource=ResourceType(perm_data["resource"]),
                        action=ActionType(perm_data["action"]),
                        conditions=perm_data.get("conditions"),
                        description=perm_data.get("description", "")
                    )
                    permissions.append(permission_rule)
                
                inherits_from = None
                if role_data.get("inherits_from"):
                    inherits_from = UserRole(role_data["inherits_from"])
                
                definition = RoleDefinition(
                    role=role,
                    name=role_data["name"],
                    description=role_data["description"],
                    permissions=permissions,
                    inherits_from=inherits_from
                )
                
                new_definitions[role] = definition
            
            # 验证成功后更新定义
            self.role_definitions = new_definitions
            self.role_hierarchy = self._build_role_hierarchy()
            self._clear_permission_cache()
            
            return True
            
        except Exception as e:
            print(f"导入角色定义失败: {e}")
            return False