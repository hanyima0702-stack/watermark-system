"""
水印提取结果数据访问对象
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, and_, func, desc
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.extraction_result import ExtractionResult


class ExtractionResultDAO(BaseDAO[ExtractionResult]):
    """水印提取结果数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, ExtractionResult)
    
    def get_model_class(self):
        return ExtractionResult
    
    def _add_relationship_loading(self, query):
        """加载提取结果关联对象"""
        return query.options(
            selectinload(ExtractionResult.file),
            selectinload(ExtractionResult.extracted_user),
            selectinload(ExtractionResult.evidence_reports)
        )
    
    async def get_file_results(self, file_id: str, limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """获取文件的提取结果"""
        return await self.find_by_filters(
            filters={'file_id': file_id},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_user_results(self, user_id: str, limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """获取用户相关的提取结果"""
        return await self.find_by_filters(
            filters={'extracted_user_id': user_id},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_high_confidence_results(self, threshold: float = 0.8,
                                        limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """获取高置信度结果"""
        async with self.get_session() as session:
            query = select(ExtractionResult).where(
                ExtractionResult.confidence_score >= threshold
            ).order_by(desc(ExtractionResult.confidence_score)).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_results_by_method(self, extraction_method: str,
                                  limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """根据提取方法获取结果"""
        return await self.find_by_filters(
            filters={'extraction_method': extraction_method},
            limit=limit,
            offset=offset,
            order_by='-created_at',
            load_relationships=True
        )
    
    async def get_recent_results(self, hours: int = 24, limit: int = 100) -> List[ExtractionResult]:
        """获取最近的提取结果"""
        since_time = datetime.utcnow() - timedelta(hours=hours)
        async with self.get_session() as session:
            query = select(ExtractionResult).where(
                ExtractionResult.created_at >= since_time
            ).order_by(desc(ExtractionResult.created_at)).limit(limit)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_valid_extractions(self, limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """获取有效的提取结果"""
        async with self.get_session() as session:
            query = select(ExtractionResult).where(
                and_(
                    ExtractionResult.extracted_user_id.isnot(None),
                    ExtractionResult.confidence_score > 0
                )
            ).order_by(desc(ExtractionResult.confidence_score)).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def search_by_confidence_range(self, min_confidence: float, max_confidence: float,
                                       limit: int = 100, offset: int = 0) -> List[ExtractionResult]:
        """根据置信度范围搜索"""
        async with self.get_session() as session:
            query = select(ExtractionResult).where(
                and_(
                    ExtractionResult.confidence_score >= min_confidence,
                    ExtractionResult.confidence_score <= max_confidence
                )
            ).order_by(desc(ExtractionResult.confidence_score)).limit(limit).offset(offset)
            
            query = self._add_relationship_loading(query)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_extraction_statistics(self, days: int = 30) -> dict:
        """获取提取统计信息"""
        async with self.get_session() as session:
            since_time = datetime.utcnow() - timedelta(days=days)
            base_query = select(ExtractionResult).where(ExtractionResult.created_at >= since_time)
            
            # 总提取次数
            total_count = await self.count({'created_at': {'gte': since_time}})
            
            # 成功提取次数
            success_query = select(func.count()).select_from(
                base_query.where(ExtractionResult.extracted_user_id.isnot(None)).subquery()
            )
            success_result = await session.execute(success_query)
            success_count = success_result.scalar()
            
            # 平均置信度
            avg_confidence_query = select(func.avg(ExtractionResult.confidence_score)).select_from(
                base_query.subquery()
            )
            avg_confidence_result = await session.execute(avg_confidence_query)
            avg_confidence = avg_confidence_result.scalar() or 0
            
            # 按方法统计
            method_query = select(
                ExtractionResult.extraction_method,
                func.count(ExtractionResult.result_id).label('count'),
                func.avg(ExtractionResult.confidence_score).label('avg_confidence')
            ).select_from(base_query.subquery()).group_by(ExtractionResult.extraction_method)
            method_result = await session.execute(method_query)
            method_stats = {
                row.extraction_method: {
                    'count': row.count,
                    'avg_confidence': float(row.avg_confidence or 0)
                }
                for row in method_result
            }
            
            return {
                'total_extractions': total_count,
                'successful_extractions': success_count,
                'success_rate': (success_count / total_count * 100) if total_count > 0 else 0,
                'avg_confidence': float(avg_confidence),
                'method_distribution': method_stats
            }