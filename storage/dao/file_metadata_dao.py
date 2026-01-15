"""
文件元数据数据访问对象
"""

from typing import List, Optional
from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload

from .base_dao import BaseDAO
from ..models.file_metadata import FileMetadata


class FileMetadataDAO(BaseDAO[FileMetadata]):
    """文件元数据数据访问对象"""
    
    def __init__(self, db_manager):
        super().__init__(db_manager, FileMetadata)
    
    def get_model_class(self):
        return FileMetadata
    
    def _add_relationship_loading(self, query):
        """加载文件关联对象"""
        return query.options(
            selectinload(FileMetadata.uploader),
            selectinload(FileMetadata.watermark_tasks),
            selectinload(FileMetadata.extraction_results)
        )
    
    async def get_by_hash(self, file_hash: str, user_id: Optional[str] = None) -> Optional[FileMetadata]:
        """根据文件哈希获取文件"""
        filters = {'file_hash': file_hash}
        if user_id:
            filters['uploaded_by'] = user_id
        
        results = await self.find_by_filters(filters, limit=1)
        return results[0] if results else None
    
    async def get_user_files(self, user_id: str, file_type: Optional[str] = None,
                           limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """获取用户文件列表"""
        filters = {'uploaded_by': user_id}
        if file_type:
            filters['file_type'] = file_type
        
        return await self.find_by_filters(
            filters=filters,
            limit=limit,
            offset=offset,
            order_by='-uploaded_at',
            load_relationships=True
        )
    
    async def get_files_by_type(self, file_type: str, limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """根据文件类型获取文件"""
        return await self.find_by_filters(
            filters={'file_type': file_type},
            limit=limit,
            offset=offset,
            order_by='-uploaded_at'
        )
    
    async def search_files(self, keyword: str, user_id: Optional[str] = None,
                         limit: int = 100, offset: int = 0) -> List[FileMetadata]:
        """搜索文件"""
        async with self.get_session() as session:
            query = select(FileMetadata).where(
                FileMetadata.original_name.ilike(f"%{keyword}%")
            )
            
            if user_id:
                query = query.where(FileMetadata.uploaded_by == user_id)
            
            query = query.order_by(FileMetadata.uploaded_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_duplicate_files(self, file_hash: str) -> List[FileMetadata]:
        """获取重复文件"""
        return await self.find_by_filters(
            filters={'file_hash': file_hash},
            order_by='-uploaded_at'
        )
    
    async def get_large_files(self, min_size_mb: int = 100, limit: int = 100) -> List[FileMetadata]:
        """获取大文件列表"""
        min_size_bytes = min_size_mb * 1024 * 1024
        async with self.get_session() as session:
            query = select(FileMetadata).where(
                FileMetadata.file_size >= min_size_bytes
            ).order_by(FileMetadata.file_size.desc()).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_file_statistics(self, user_id: Optional[str] = None) -> dict:
        """获取文件统计信息"""
        async with self.get_session() as session:
            base_query = select(FileMetadata)
            if user_id:
                base_query = base_query.where(FileMetadata.uploaded_by == user_id)
            
            # 总文件数
            total_count = await self.count({'uploaded_by': user_id} if user_id else None)
            
            # 总文件大小
            size_query = select(func.sum(FileMetadata.file_size)).select_from(base_query.subquery())
            size_result = await session.execute(size_query)
            total_size = size_result.scalar() or 0
            
            # 按文件类型统计
            type_query = select(
                FileMetadata.file_type,
                func.count(FileMetadata.file_id).label('count'),
                func.sum(FileMetadata.file_size).label('size')
            ).select_from(base_query.subquery()).group_by(FileMetadata.file_type)
            type_result = await session.execute(type_query)
            type_stats = {
                row.file_type: {'count': row.count, 'size': row.size}
                for row in type_result
            }
            
            return {
                'total_files': total_count,
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'type_distribution': type_stats
            }