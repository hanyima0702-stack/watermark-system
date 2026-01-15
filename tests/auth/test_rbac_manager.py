"""
RBAC管理器测试
"""

import pytest
from business.auth.rbac_manager import RBACManager, ResourceType, ActionType, PermissionRule
from business.auth.models import UserRole


class TestRBACManager:
    """RBAC管理器测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.rbac_manager = RBACManager()
    
    def test_admin_has_all_permissions(self):
        """测试管理员拥有所有权限"""
        admin_roles = [UserRole.ADMIN]
        
        # 测试各种资源和操作
        test_cases = [
            ("file", "create"),
            ("file", "read"),
            ("file", "update"),
            ("file", "delete"),
            ("watermark", "create"),
            ("watermark", "execute"),
            ("user", "manage"),
            ("system", "update"),
            ("audit", "read"),
            ("report", "create")
        ]
        
        for resource, action in test_cases:
            assert self.rbac_manager.check_permission(admin_roles, resource, action)
    
    def test_user_basic_permissions(self):
        """测试普通用户基本权限"""
        user_roles = [UserRole.USER]
        
        # 用户应该有的权限
        allowed_permissions = [
            ("file", "create"),
            ("watermark", "create"),
        ]
        
        for resource, action in allowed_permissions:
            assert self.rbac_manager.check_permission(user_roles, resource, action)
        
        # 用户不应该有的权限
        denied_permissions = [
            ("file", "delete"),
            ("user", "manage"),
            ("system", "update"),
            ("audit", "read")
        ]
        
        for resource, action in denied_permissions:
            assert not self.rbac_manager.check_permission(user_roles, resource, action)
    
    def test_operator_permissions(self):
        """测试操作员权限"""
        operator_roles = [UserRole.OPERATOR]
        
        # 操作员应该有的权限
        allowed_permissions = [
            ("file", "create"),
            ("file", "read"),
            ("file", "update"),
            ("watermark", "create"),
            ("watermark", "read"),
            ("watermark", "update"),
            ("watermark", "execute"),
            ("report", "create"),
            ("report", "read")
        ]
        
        for resource, action in allowed_permissions:
            assert self.rbac_manager.check_permission(operator_roles, resource, action)
        
        # 操作员不应该有的权限
        denied_permissions = [
            ("user", "manage"),
            ("system", "update"),
            ("audit", "read")
        ]
        
        for resource, action in denied_permissions:
            assert not self.rbac_manager.check_permission(operator_roles, resource, action)
    
    def test_auditor_permissions(self):
        """测试审计员权限"""
        auditor_roles = [UserRole.AUDITOR]
        
        # 审计员应该有的权限
        allowed_permissions = [
            ("file", "read"),
            ("watermark", "execute"),
            ("audit", "read"),
            ("report", "read")
        ]
        
        for resource, action in allowed_permissions:
            assert self.rbac_manager.check_permission(auditor_roles, resource, action)
        
        # 审计员不应该有的权限
        denied_permissions = [
            ("file", "create"),
            ("file", "update"),
            ("file", "delete"),
            ("watermark", "create"),
            ("user", "manage"),
            ("system", "update")
        ]
        
        for resource, action in denied_permissions:
            assert not self.rbac_manager.check_permission(auditor_roles, resource, action)
    
    def test_role_inheritance(self):
        """测试角色继承"""
        # 操作员继承用户权限
        operator_roles = [UserRole.OPERATOR]
        
        # 检查继承的用户权限
        user_permissions = [
            ("file", "create"),
            ("watermark", "create")
        ]
        
        for resource, action in user_permissions:
            assert self.rbac_manager.check_permission(operator_roles, resource, action)
    
    def test_multiple_roles(self):
        """测试多角色权限"""
        multiple_roles = [UserRole.USER, UserRole.AUDITOR]
        
        # 应该拥有两个角色的所有权限
        combined_permissions = [
            ("file", "create"),  # 来自USER角色
            ("file", "read"),    # 来自AUDITOR角色
            ("watermark", "create"),  # 来自USER角色
            ("watermark", "execute"), # 来自AUDITOR角色
            ("audit", "read"),   # 来自AUDITOR角色
            ("report", "read")   # 来自AUDITOR角色
        ]
        
        for resource, action in combined_permissions:
            assert self.rbac_manager.check_permission(multiple_roles, resource, action)
    
    def test_permission_conditions_owner_check(self):
        """测试权限条件检查（所有者检查）"""
        user_roles = [UserRole.USER]
        
        # 测试所有者权限
        owner_context = {
            "current_user": "user123",
            "resource_owner": "user123",
            "owner": True
        }
        
        non_owner_context = {
            "current_user": "user123",
            "resource_owner": "user456",
            "owner": True
        }
        
        # 用户应该能访问自己的文件
        assert self.rbac_manager.check_permission(
            user_roles, "file", "read", owner_context
        )
        
        # 用户不应该能访问别人的文件（如果有所有者限制）
        # 注意：这取决于具体的权限规则定义
    
    def test_get_user_permissions(self):
        """测试获取用户权限列表"""
        user_roles = [UserRole.USER]
        permissions = self.rbac_manager.get_user_permissions(user_roles)
        
        assert isinstance(permissions, list)
        assert len(permissions) > 0
        
        # 检查权限格式
        for permission in permissions:
            assert ":" in permission
            resource, action = permission.split(":")
            assert resource in [rt.value for rt in ResourceType]
            assert action in [at.value for at in ActionType]
    
    def test_get_role_permissions(self):
        """测试获取角色权限"""
        user_permissions = self.rbac_manager.get_role_permissions(UserRole.USER)
        admin_permissions = self.rbac_manager.get_role_permissions(UserRole.ADMIN)
        
        assert isinstance(user_permissions, list)
        assert isinstance(admin_permissions, list)
        assert len(admin_permissions) > len(user_permissions)
        
        # 检查权限规则结构
        for permission_rule in user_permissions:
            assert isinstance(permission_rule, PermissionRule)
            assert isinstance(permission_rule.resource, ResourceType)
            assert isinstance(permission_rule.action, ActionType)
    
    def test_get_all_roles(self):
        """测试获取所有角色定义"""
        all_roles = self.rbac_manager.get_all_roles()
        
        assert isinstance(all_roles, list)
        assert len(all_roles) == 4  # USER, OPERATOR, AUDITOR, ADMIN
        
        role_names = [role_def.role for role_def in all_roles]
        expected_roles = [UserRole.USER, UserRole.OPERATOR, UserRole.AUDITOR, UserRole.ADMIN]
        
        for expected_role in expected_roles:
            assert expected_role in role_names
    
    def test_add_role_permission(self):
        """测试添加角色权限"""
        new_permission = PermissionRule(
            resource=ResourceType.FILE,
            action=ActionType.EXECUTE,
            description="执行文件操作"
        )
        
        # 添加权限
        result = self.rbac_manager.add_role_permission(UserRole.USER, new_permission)
        assert result is True
        
        # 验证权限已添加
        user_permissions = self.rbac_manager.get_role_permissions(UserRole.USER)
        permission_exists = any(
            p.resource == ResourceType.FILE and p.action == ActionType.EXECUTE
            for p in user_permissions
        )
        assert permission_exists
        
        # 尝试添加重复权限
        duplicate_result = self.rbac_manager.add_role_permission(UserRole.USER, new_permission)
        assert duplicate_result is False
    
    def test_remove_role_permission(self):
        """测试移除角色权限"""
        # 先添加一个权限
        new_permission = PermissionRule(
            resource=ResourceType.CONFIG,
            action=ActionType.DELETE,
            description="删除配置"
        )
        self.rbac_manager.add_role_permission(UserRole.USER, new_permission)
        
        # 移除权限
        result = self.rbac_manager.remove_role_permission(
            UserRole.USER, 
            ResourceType.CONFIG, 
            ActionType.DELETE
        )
        assert result is True
        
        # 验证权限已移除
        user_permissions = self.rbac_manager.get_role_permissions(UserRole.USER)
        permission_exists = any(
            p.resource == ResourceType.CONFIG and p.action == ActionType.DELETE
            for p in user_permissions
        )
        assert not permission_exists
        
        # 尝试移除不存在的权限
        non_existent_result = self.rbac_manager.remove_role_permission(
            UserRole.USER,
            ResourceType.SYSTEM,
            ActionType.DELETE
        )
        assert non_existent_result is False
    
    def test_validate_permission_string(self):
        """测试权限字符串验证"""
        # 有效的权限字符串
        valid_permissions = [
            "file:create",
            "watermark:read",
            "user:manage",
            "system:update"
        ]
        
        for permission in valid_permissions:
            assert self.rbac_manager.validate_permission_string(permission)
        
        # 无效的权限字符串
        invalid_permissions = [
            "invalid_format",
            "file:",
            ":create",
            "file:invalid_action",
            "invalid_resource:create",
            "file:create:extra"
        ]
        
        for permission in invalid_permissions:
            assert not self.rbac_manager.validate_permission_string(permission)
    
    def test_get_permission_description(self):
        """测试获取权限描述"""
        description = self.rbac_manager.get_permission_description("file:create")
        assert description is not None
        assert isinstance(description, str)
        
        # 测试无效权限
        invalid_description = self.rbac_manager.get_permission_description("invalid:permission")
        assert invalid_description is None
    
    def test_check_role_hierarchy(self):
        """测试角色继承关系检查"""
        # 操作员继承用户角色
        assert self.rbac_manager.check_role_hierarchy(UserRole.USER, UserRole.OPERATOR)
        
        # 用户不继承操作员角色
        assert not self.rbac_manager.check_role_hierarchy(UserRole.OPERATOR, UserRole.USER)
        
        # 管理员不继承其他角色（在当前设计中）
        assert not self.rbac_manager.check_role_hierarchy(UserRole.USER, UserRole.ADMIN)
    
    def test_get_role_hierarchy_tree(self):
        """测试获取角色继承树"""
        hierarchy_tree = self.rbac_manager.get_role_hierarchy_tree()
        
        assert isinstance(hierarchy_tree, dict)
        assert len(hierarchy_tree) == 4
        
        # 检查每个角色的信息
        for role_value, role_info in hierarchy_tree.items():
            assert "name" in role_info
            assert "description" in role_info
            assert "permissions_count" in role_info
            assert isinstance(role_info["permissions_count"], int)
    
    def test_export_import_role_definitions(self):
        """测试角色定义的导出和导入"""
        # 导出角色定义
        exported_data = self.rbac_manager.export_role_definitions()
        
        assert isinstance(exported_data, dict)
        assert len(exported_data) == 4
        
        # 检查导出数据结构
        for role_name, role_data in exported_data.items():
            assert "name" in role_data
            assert "description" in role_data
            assert "permissions" in role_data
            assert isinstance(role_data["permissions"], list)
        
        # 导入角色定义
        import_result = self.rbac_manager.import_role_definitions(exported_data)
        assert import_result is True
        
        # 验证导入后的数据
        imported_roles = self.rbac_manager.get_all_roles()
        assert len(imported_roles) == 4
    
    def test_invalid_resource_action(self):
        """测试无效的资源和操作"""
        user_roles = [UserRole.USER]
        
        # 测试无效资源
        assert not self.rbac_manager.check_permission(user_roles, "invalid_resource", "read")
        
        # 测试无效操作
        assert not self.rbac_manager.check_permission(user_roles, "file", "invalid_action")
    
    def test_empty_roles(self):
        """测试空角色列表"""
        empty_roles = []
        
        # 空角色不应该有任何权限
        assert not self.rbac_manager.check_permission(empty_roles, "file", "read")
        
        # 获取空角色的权限应该返回空列表
        permissions = self.rbac_manager.get_user_permissions(empty_roles)
        assert permissions == []