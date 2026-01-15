"""
数据库管理器
提供数据库连接、初始化和管理功能
"""

import asyncio
import logging
from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager

from .models import Base, User, SystemConfig
from .dao import (
    UserDAO, FileMetadataDAO, WatermarkConfigDAO, WatermarkTaskDAO,
    ExtractionResultDAO, EvidenceReportDAO, AuditLogDAO, 
    SystemConfigDAO, KeyManagementDAO
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        self._daos = {}
        
    async def initialize(self):
        """初始化数据库连接"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 初始化DAO对象
            self._init_daos()
            
            logger.info("数据库连接初始化成功")
            
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise
    
    def _init_daos(self):
        """初始化DAO对象"""
        self._daos = {
            'user': UserDAO(self),
            'file_metadata': FileMetadataDAO(self),
            'watermark_config': WatermarkConfigDAO(self),
            'watermark_task': WatermarkTaskDAO(self),
            'extraction_result': ExtractionResultDAO(self),
            'evidence_report': EvidenceReportDAO(self),
            'audit_log': AuditLogDAO(self),
            'system_config': SystemConfigDAO(self),
            'key_management': KeyManagementDAO(self)
        }
    
    @asynccontextmanager
    async def get_session(self):
        """获取数据库会话"""
        if not self.async_session:
            raise RuntimeError("数据库未初始化，请先调用initialize()")
        
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_tables(self):
        """创建数据表"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("数据表创建成功")
        except SQLAlchemyError as e:
            logger.error(f"数据表创建失败: {e}")
            raise
    
    async def drop_tables(self):
        """删除数据表"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("数据表删除成功")
        except SQLAlchemyError as e:
            logger.error(f"数据表删除失败: {e}")
            raise
    
    async def init_default_data(self):
        """初始化默认数据"""
        try:
            # 创建默认管理员用户
            await self._create_default_admin()
            
            # 创建默认系统配置
            await self._create_default_configs()
            
            logger.info("默认数据初始化成功")
            
        except Exception as e:
            logger.error(f"默认数据初始化失败: {e}")
            raise
    
    async def _create_default_admin(self):
        """创建默认管理员用户"""
        user_dao = self.get_dao('user')
        
        # 检查是否已存在管理员用户
        admin_user = await user_dao.get_by_username('administrator')
        if admin_user:
            logger.info("管理员用户已存在，跳过创建")
            return
        
        # 创建默认管理员
        admin = User(
            user_id='admin',
            username='administrator',
            email='admin@watermark-system.com',
            department='IT',
            roles=['admin', 'operator'],
            password_hash='5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',
            salt='default_salt',
            is_active=True
        )
        
        await user_dao.create(admin)
        logger.info("默认管理员用户创建成功")
    
    async def _create_default_configs(self):
        """创建默认系统配置"""
        config_dao = self.get_dao('system_config')
        
        default_configs = SystemConfig.get_default_configs()
        
        for key, config_data in default_configs.items():
            # 检查配置是否已存在
            existing_config = await config_dao.get_by_id(key)
            if existing_config:
                continue
            
            # 创建新配置
            config = SystemConfig.create_config(
                key=key,
                value=config_data['value'],
                description=config_data['description'],
                is_encrypted=config_data.get('encrypted', False)
            )
            
            await config_dao.create(config)
        
        logger.info("默认系统配置创建成功")
    
    async def health_check(self) -> bool:
        """数据库健康检查"""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭数据库连接"""
        if self.engine:
            await self.engine.dispose()
            logger.info("数据库连接已关闭")
    
    def get_dao(self, dao_name: str):
        """获取DAO对象"""
        if dao_name not in self._daos:
            raise ValueError(f"未知的DAO名称: {dao_name}")
        return self._daos[dao_name]
    
    @property
    def user_dao(self) -> UserDAO:
        return self.get_dao('user')
    
    @property
    def file_metadata_dao(self) -> FileMetadataDAO:
        return self.get_dao('file_metadata')
    
    @property
    def watermark_config_dao(self) -> WatermarkConfigDAO:
        return self.get_dao('watermark_config')
    
    @property
    def watermark_task_dao(self) -> WatermarkTaskDAO:
        return self.get_dao('watermark_task')
    
    @property
    def extraction_result_dao(self) -> ExtractionResultDAO:
        return self.get_dao('extraction_result')
    
    @property
    def evidence_report_dao(self) -> EvidenceReportDAO:
        return self.get_dao('evidence_report')
    
    @property
    def audit_log_dao(self) -> AuditLogDAO:
        return self.get_dao('audit_log')
    
    @property
    def system_config_dao(self) -> SystemConfigDAO:
        return self.get_dao('system_config')
    
    @property
    def key_management_dao(self) -> KeyManagementDAO:
        return self.get_dao('key_management')


# 全局数据库管理器实例
db_manager: Optional[DatabaseManager] = None


async def init_database(database_url: str) -> DatabaseManager:
    """初始化数据库"""
    global db_manager
    
    db_manager = DatabaseManager(database_url)
    await db_manager.initialize()
    
    return db_manager


async def get_database() -> DatabaseManager:
    """获取数据库管理器实例"""
    if not db_manager:
        raise RuntimeError("数据库未初始化，请先调用init_database()")
    return db_manager


async def close_database():
    """关闭数据库连接"""
    global db_manager
    if db_manager:
        await db_manager.close()
        db_manager = None


# 便捷函数
async def setup_database(database_url: str, create_tables: bool = True, init_data: bool = True):
    """设置数据库（创建表和初始化数据）"""
    db = await init_database(database_url)
    
    if create_tables:
        await db.create_tables()
    
    if init_data:
        await db.init_default_data()
    
    return db


if __name__ == "__main__":
    # 测试脚本
    async def main():
        database_url = "postgresql+asyncpg://watermark_user:watermark_pass@localhost:5432/watermark_system"
        
        try:
            db = await setup_database(database_url)
            
            # 健康检查
            if await db.health_check():
                print("数据库连接正常")
            else:
                print("数据库连接异常")
            
            await close_database()
            
        except Exception as e:
            print(f"数据库设置失败: {e}")
    
    asyncio.run(main())