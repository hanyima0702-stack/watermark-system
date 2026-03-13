"""
用户会话数据访问对象
提供用户会话相关的数据库操作
"""

from typing import List, Optional
from sqlalchemy import select, update, and_
from datetime import datetime

from .base_dao import BaseDAO
from ..models.user_session import UserSession


class UserSessionDAO(BaseDAO[UserSession]):
    """用户会话数据访问对象"""

    def __init__(self, db_manager):
        super().__init__(db_manager, UserSession)

    def get_model_class(self):
        return UserSession

    async def get_active_sessions_by_user(self, user_id: str) -> List[UserSession]:
        """获取用户的活跃会话列表"""
        async with self.get_session() as session:
            query = select(UserSession).where(
                and_(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            ).order_by(UserSession.created_at.desc())
            result = await session.execute(query)
            return list(result.scalars().all())

    async def deactivate_by_token_hash(self, token_hash: str) -> bool:
        """根据token_hash将会话标记为非活跃"""
        async with self.get_session() as session:
            stmt = (
                update(UserSession)
                .where(UserSession.token_hash == token_hash)
                .values(is_active=False)
            )
            result = await session.execute(stmt)
            return result.rowcount > 0

    async def deactivate_all_user_sessions(self, user_id: str) -> int:
        """将用户所有会话标记为非活跃"""
        async with self.get_session() as session:
            stmt = (
                update(UserSession)
                .where(
                    and_(
                        UserSession.user_id == user_id,
                        UserSession.is_active == True
                    )
                )
                .values(is_active=False)
            )
            result = await session.execute(stmt)
            return result.rowcount
