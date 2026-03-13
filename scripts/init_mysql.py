#!/usr/bin/env python3
"""
MySQL数据库初始化脚本
执行所有SQL迁移脚本，创建数据库表结构
Requirements: 1.5
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiomysql
from shared.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MySQLInitializer:
    """MySQL数据库初始化器"""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.migrations_dir = project_root / "storage" / "migrations" / "mysql"
        
    async def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = await aiomysql.connect(
                host=self.config.database.host,
                port=self.config.database.port,
                user=self.config.database.user,
                password=self.config.database.password,
                db=self.config.database.database,
                charset=self.config.database.charset,
                autocommit=False
            )
            logger.info(f"成功连接到MySQL数据库: {self.config.database.host}:{self.config.database.port}/{self.config.database.database}")
            return True
        except Exception as e:
            logger.error(f"连接MySQL数据库失败: {e}")
            return False
    
    async def test_connection(self):
        """测试数据库连接"""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute("SELECT VERSION()")
                version = await cursor.fetchone()
                logger.info(f"MySQL版本: {version[0]}")
                
                await cursor.execute("SELECT DATABASE()")
                database = await cursor.fetchone()
                logger.info(f"当前数据库: {database[0]}")
                
            return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def get_migration_files(self) -> List[Path]:
        """获取所有迁移文件，按文件名排序"""
        if not self.migrations_dir.exists():
            logger.error(f"迁移目录不存在: {self.migrations_dir}")
            return []
        
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        logger.info(f"找到 {len(migration_files)} 个迁移文件")
        return migration_files
    
    async def execute_migration(self, migration_file: Path) -> bool:
        """执行单个迁移文件"""
        try:
            logger.info(f"执行迁移: {migration_file.name}")
            
            # 读取SQL文件内容
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 移除注释和空行，分割SQL语句
            sql_statements = []
            for line in sql_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('--'):
                    sql_statements.append(line)
            
            sql = ' '.join(sql_statements)
            
            # 执行SQL
            async with self.connection.cursor() as cursor:
                await cursor.execute(sql)
                await self.connection.commit()
            
            logger.info(f"✓ 迁移成功: {migration_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ 迁移失败: {migration_file.name} - {e}")
            await self.connection.rollback()
            return False
    
    async def verify_tables(self) -> bool:
        """验证表是否创建成功"""
        expected_tables = ['users', 'files', 'watermark_tasks', 'user_sessions']
        
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute("SHOW TABLES")
                tables = await cursor.fetchall()
                existing_tables = [table[0] for table in tables]
                
                logger.info(f"数据库中的表: {existing_tables}")
                
                missing_tables = [t for t in expected_tables if t not in existing_tables]
                
                if missing_tables:
                    logger.error(f"缺少以下表: {missing_tables}")
                    return False
                
                logger.info("✓ 所有必需的表都已创建")
                
                # 显示每个表的结构
                for table in expected_tables:
                    await cursor.execute(f"DESCRIBE {table}")
                    columns = await cursor.fetchall()
                    logger.info(f"\n表 {table} 的结构:")
                    for col in columns:
                        logger.info(f"  - {col[0]} ({col[1]})")
                
                return True
                
        except Exception as e:
            logger.error(f"验证表结构失败: {e}")
            return False
    
    async def run_migrations(self) -> bool:
        """运行所有迁移"""
        migration_files = self.get_migration_files()
        
        if not migration_files:
            logger.error("没有找到迁移文件")
            return False
        
        success_count = 0
        for migration_file in migration_files:
            if await self.execute_migration(migration_file):
                success_count += 1
            else:
                logger.error(f"迁移失败，停止执行")
                return False
        
        logger.info(f"\n成功执行 {success_count}/{len(migration_files)} 个迁移")
        return success_count == len(migration_files)
    
    async def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("MySQL数据库初始化脚本")
    logger.info("=" * 60)
    
    # 加载配置
    settings = get_settings()
    
    # 显示配置信息
    logger.info(f"\n数据库配置:")
    logger.info(f"  类型: {settings.database.db_type}")
    logger.info(f"  主机: {settings.database.host}")
    logger.info(f"  端口: {settings.database.port}")
    logger.info(f"  数据库: {settings.database.database}")
    logger.info(f"  用户: {settings.database.user}")
    logger.info(f"  字符集: {settings.database.charset}")
    
    # 创建初始化器
    initializer = MySQLInitializer(settings)
    
    try:
        # 1. 连接数据库
        logger.info("\n步骤 1: 连接数据库")
        if not await initializer.connect():
            logger.error("无法连接到数据库，退出")
            return False
        
        # 2. 测试连接
        logger.info("\n步骤 2: 测试数据库连接")
        if not await initializer.test_connection():
            logger.error("数据库连接测试失败，退出")
            return False
        
        # 3. 执行迁移
        logger.info("\n步骤 3: 执行数据库迁移")
        if not await initializer.run_migrations():
            logger.error("数据库迁移失败，退出")
            return False
        
        # 4. 验证表结构
        logger.info("\n步骤 4: 验证表结构")
        if not await initializer.verify_tables():
            logger.error("表结构验证失败")
            return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ 数据库初始化成功！")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"\n数据库初始化过程中发生错误: {e}")
        return False
        
    finally:
        await initializer.close()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
