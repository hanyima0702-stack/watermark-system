"""
测试MySQL数据库连接
用于验证MySQL配置是否正确
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import get_settings
from storage.database_manager import DatabaseManager


async def test_mysql_connection():
    """测试MySQL连接"""
    print("=" * 60)
    print("MySQL数据库连接测试")
    print("=" * 60)
    
    # 获取配置
    settings = get_settings()
    db_config = settings.database
    
    print(f"\n数据库配置:")
    print(f"  类型: {db_config.db_type}")
    print(f"  主机: {db_config.host}")
    print(f"  端口: {db_config.port}")
    print(f"  数据库: {db_config.database}")
    print(f"  用户: {db_config.user}")
    print(f"  字符集: {db_config.charset}")
    print(f"  连接池大小: {db_config.pool_size}")
    print(f"  最大溢出: {db_config.max_overflow}")
    
    # 生成连接URL (隐藏密码)
    connection_url = db_config.connection_url
    safe_url = connection_url.replace(db_config.password, "****")
    print(f"\n连接URL: {safe_url}")
    
    # 创建数据库管理器
    print("\n正在初始化数据库连接...")
    db_manager = DatabaseManager(
        database_url=connection_url,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow
    )
    
    try:
        # 初始化连接
        await db_manager.initialize()
        print("✓ 数据库连接初始化成功")
        
        # 执行健康检查
        print("\n正在执行健康检查...")
        is_healthy = await db_manager.health_check()
        
        if is_healthy:
            print("✓ 数据库健康检查通过")
            print("\n数据库连接测试成功!")
            return True
        else:
            print("✗ 数据库健康检查失败")
            return False
            
    except Exception as e:
        print(f"\n✗ 数据库连接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 关闭连接
        await db_manager.close()
        print("\n数据库连接已关闭")


async def main():
    """主函数"""
    success = await test_mysql_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
