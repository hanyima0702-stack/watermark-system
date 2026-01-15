"""
用户数据访问对象
提供用户相关的数据库操作
"""

from typing import List, Optional
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.user import User


class UserDAO(BaseDAO[User]):
    """用户数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, User)
    
    def get_model_class(self):
        return User
    
    def _add_relationship_loading(self, query):
        """加载用户关联对象"""
        return query.options(
            selectinload(User.watermark_tasks),
            selectinload(User.uploaded_files),
            selectinload(User.watermark_configs)
        )
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        async with self.get_session() as session:
            query = select(User).where(User.username == username)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        async with self.get_session() as session:
            query = select(User).where(User.email == email)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def get_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """获取活跃用户列表"""
        return await self.find_by_filters(
            filters={'is_active': True},
            limit=limit,
            offset=offset,
            order_by='created_at'
        )
    
    async def get_users_by_role(self, role: str, limit: int = 100, offset: int = 0) -> List[User]:
        """根据角色获取用户"""
        async with self.get_session() as session:
            query = select(User).where(
                User.roles.any(role)
            ).limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_users_by_department(self, department: str, limit: int = 100, offset: int = 0) -> List[User]:
        """根据部门获取用户"""
        return await self.find_by_filters(
            filters={'department': department, 'is_active': True},
            limit=limit,
            offset=offset,
            order_by='username'
        )
    
    async def search_users(self, keyword: str, limit: int = 100, offset: int = 0) -> List[User]:
        """搜索用户"""
        async with self.get_session() as session:
            query = select(User).where(
                and_(
                    User.is_active == True,
                    or_(
                        User.username.ilike(f"%{keyword}%"),
                        User.email.ilike(f"%{keyword}%"),
                        User.department.ilike(f"%{keyword}%")
                    )
                )
            ).limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def activate_user(self, user_id: str) -> bool:
        """激活用户"""
        result = await self.update(user_id, {'is_active': True})
        return result is not None
    
    async def deactivate_user(self, user_id: str) -> bool:
        """停用用户"""
        result = await self.update(user_id, {'is_active': False})
        return result is not None
    
    async def update_password(self, user_id: str, password_hash: str, salt: str) -> bool:
        """更新用户密码"""
        result = await self.update(user_id, {
            'password_hash': password_hash,
            'salt': salt
        })
        return result is not None
    
    async def add_user_role(self, user_id: str, role: str) -> bool:
        """添加用户角色"""
        user = await self.get_by_id(user_id)
        if user:
            user.add_role(role)
            result = await self.update(user_id, {'roles': user.roles})
            return result is not None
        return False
    
    async def remove_user_role(self, user_id: str, role: str) -> bool:
        """移除用户角色"""
        user = await self.get_by_id(user_id)
        if user:
            user.remove_role(role)
            result = await self.update(user_id, {'roles': user.roles})
            return result is not None
        return False
    
    async def get_user_statistics(self) -> dict:
        """获取用户统计信息"""
        async with self.get_session() as session:
            # 总用户数
            total_users = await self.count()
            
            # 活跃用户数
            active_users = await self.count({'is_active': True})
            
            # 按角色统计
            admin_count = len(await self.get_users_by_role('admin'))
            operator_count = len(await self.get_users_by_role('operator'))
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': total_users - active_users,
                'admin_count': admin_count,
                'operator_count': operator_count
            }