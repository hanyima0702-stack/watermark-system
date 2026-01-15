"""
证据报告数据访问对象
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.evidence_report import EvidenceReport


class EvidenceReportDAO(BaseDAO[EvidenceReport]):
    """证据报告数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, EvidenceReport)
    
    def get_model_class(self):
        return EvidenceReport
    
    def _add_relationship_loading(self, query):
        """加载报告关联对象"""
        return query.options(
            selectinload(EvidenceReport.extraction_result),
            selectinload(EvidenceReport.generator)
        )
    
    async def get_user_reports(self, user_id: str, limit: int = 100, offset: int = 0) -> List[EvidenceReport]:
        """获取用户生成的报告"""
        return await self.find_by_filters(
            filters={'generated_by': user_id},
            limit=limit,
            offset=offset,
            order_by='-generated_at',
            load_relationships=True
        )
    
    async def get_reports_by_result(self, result_id: str) -> List[EvidenceReport]:
        """获取提取结果相关的报告"""
        return await self.find_by_filters(
            filters={'result_id': result_id},
            order_by='-generated_at',
            load_relationships=True
        )
    
    async def get_recent_reports(self, days: int = 7, limit: int = 100) -> List[EvidenceReport]:
        """获取最近的报告"""
        since_time = datetime.utcnow() - timedelta(days=days)
        async with self.get_session() as session:
            query = select(EvidenceReport).where(
                EvidenceReport.generated_at >= since_time
            ).order_by(EvidenceReport.generated_at.desc()).limit(limit)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_signed_reports(self, limit: int = 100, offset: int = 0) -> List[EvidenceReport]:
        """获取已签名的报告"""
        async with self.get_session() as session:
            query = select(EvidenceReport).where(
                EvidenceReport.report_metadata['is_signed'].astext.cast(Boolean) == True
            ).order_by(EvidenceReport.generated_at.desc()).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def search_reports(self, keyword: str, user_id: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> List[EvidenceReport]:
        """搜索报告"""
        async with self.get_session() as session:
            query = select(EvidenceReport).where(
                EvidenceReport.report_path.ilike(f"%{keyword}%")
            )
            
            if user_id:
                query = query.where(EvidenceReport.generated_by == user_id)
            
            query = query.order_by(EvidenceReport.generated_at.desc()).limit(limit).offset(offset)
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()