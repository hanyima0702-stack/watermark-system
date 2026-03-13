# MySQL数据库配置指南

## 概述

本文档说明如何配置水印系统使用MySQL数据库。系统已经集成了MySQL支持,包括异步连接、连接池管理和自动表创建功能。

## 依赖包

系统已在 `requirements.txt` 中包含以下MySQL相关依赖:

```
aiomysql==0.2.0      # MySQL异步驱动
pymysql==1.1.0       # MySQL同步驱动(作为备用)
```

安装依赖:

```bash
pip install -r requirements.txt
```

## 配置方式

### 方式1: 使用环境变量(推荐)

在 `.env` 文件中配置MySQL连接参数:

```bash
# 数据库类型
DATABASE_TYPE=mysql

# MySQL连接参数
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=watermark_user
DATABASE_PASSWORD=watermark_pass
DATABASE_DATABASE=watermark_system
DATABASE_CHARSET=utf8mb4

# 连接池配置
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

### 方式2: 使用完整连接URL

也可以直接使用完整的连接URL:

```bash
DATABASE_URL=mysql+aiomysql://watermark_user:watermark_pass@localhost:3306/watermark_system?charset=utf8mb4
```

**注意**: 如果设置了 `DATABASE_URL`,系统将忽略单独的连接参数配置。

## MySQL数据库准备

### 1. 创建数据库

```sql
CREATE DATABASE watermark_system 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;
```

### 2. 创建用户并授权

```sql
CREATE USER 'watermark_user'@'localhost' IDENTIFIED BY 'watermark_pass';
GRANT ALL PRIVILEGES ON watermark_system.* TO 'watermark_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 验证连接

```bash
mysql -u watermark_user -p watermark_system
```

## 配置说明

### DatabaseConfig 类

系统使用 `shared/config.py` 中的 `DatabaseConfig` 类管理数据库配置:

```python
class DatabaseConfig(BaseSettings):
    db_type: str = "mysql"           # 数据库类型
    host: str = "localhost"          # 主机地址
    port: int = 3306                 # 端口号
    user: str = "watermark_user"     # 用户名
    password: str = "watermark_pass" # 密码
    database: str = "watermark_system" # 数据库名
    charset: str = "utf8mb4"         # 字符集
    pool_size: int = 20              # 连接池大小
    max_overflow: int = 30           # 最大溢出连接数
```

### 连接URL生成

系统会自动根据配置生成正确的连接URL:

```python
# MySQL连接URL格式
mysql+aiomysql://用户名:密码@主机:端口/数据库?charset=字符集
```

## 使用示例

### 初始化数据库连接

```python
from shared.config import get_settings
from storage.database_manager import init_database

# 获取配置
settings = get_settings()
database_url = settings.database.connection_url

# 初始化数据库
db_manager = await init_database(database_url)
```

### 创建数据表

```python
# 创建所有表
await db_manager.create_tables()

# 初始化默认数据
await db_manager.init_default_data()
```

### 使用数据库会话

```python
# 使用上下文管理器
async with db_manager.get_session() as session:
    # 执行数据库操作
    result = await session.execute(query)
    # 自动提交或回滚
```

## 测试连接

运行测试脚本验证MySQL配置:

```bash
cd watermark-system
python scripts/test_mysql_connection.py
```

测试脚本会:
1. 读取配置文件
2. 初始化数据库连接
3. 执行健康检查
4. 显示连接状态

## 连接池配置

### 推荐配置

- **开发环境**: pool_size=5, max_overflow=10
- **生产环境**: pool_size=20, max_overflow=30
- **高负载环境**: pool_size=50, max_overflow=100

### 配置参数说明

- `pool_size`: 连接池中保持的连接数
- `max_overflow`: 超过pool_size后允许创建的额外连接数
- `pool_pre_ping`: 使用连接前先ping测试(已启用)
- `pool_recycle`: 连接回收时间(3600秒)

## 字符集配置

系统默认使用 `utf8mb4` 字符集,支持:
- 完整的Unicode字符
- Emoji表情符号
- 多语言文本

确保MySQL服务器配置:

```ini
[mysqld]
character-set-server=utf8mb4
collation-server=utf8mb4_unicode_ci
```

## 性能优化建议

### 1. MySQL服务器配置

```ini
[mysqld]
# 连接数
max_connections=500

# InnoDB缓冲池
innodb_buffer_pool_size=2G

# 查询缓存
query_cache_size=64M
query_cache_type=1

# 日志配置
slow_query_log=1
long_query_time=2
```

### 2. 索引优化

系统会自动创建必要的索引,包括:
- 主键索引
- 外键索引
- 唯一索引(username, email)
- 查询优化索引

### 3. 连接池监控

```python
# 获取连接池状态
pool = db_manager.engine.pool
print(f"连接池大小: {pool.size()}")
print(f"已签出连接: {pool.checkedout()}")
print(f"溢出连接: {pool.overflow()}")
```

## 故障排查

### 连接失败

**错误**: `Can't connect to MySQL server`

**解决方案**:
1. 检查MySQL服务是否运行
2. 验证主机和端口配置
3. 检查防火墙设置
4. 确认用户名和密码正确

### 字符集问题

**错误**: `Incorrect string value`

**解决方案**:
1. 确保数据库使用utf8mb4字符集
2. 检查表和列的字符集设置
3. 在连接URL中指定charset参数

### 连接池耗尽

**错误**: `QueuePool limit exceeded`

**解决方案**:
1. 增加pool_size和max_overflow
2. 检查是否有连接泄漏
3. 确保正确使用上下文管理器
4. 优化长时间运行的查询

## 从PostgreSQL迁移

如果从PostgreSQL迁移到MySQL,注意以下差异:

### 1. 数据类型映射

| PostgreSQL | MySQL |
|------------|-------|
| SERIAL | INT AUTO_INCREMENT |
| TEXT | TEXT |
| TIMESTAMP | DATETIME |
| BOOLEAN | TINYINT(1) |
| UUID | CHAR(36) |

### 2. SQL语法差异

- PostgreSQL使用 `RETURNING *`,MySQL使用 `LAST_INSERT_ID()`
- PostgreSQL使用 `LIMIT/OFFSET`,MySQL也支持但性能不同
- 日期函数名称可能不同

### 3. 迁移脚本

使用提供的迁移脚本:

```bash
python scripts/migrate_to_mysql.py
```

## 安全建议

1. **不要在代码中硬编码密码**
   - 使用环境变量
   - 使用密钥管理服务

2. **使用强密码**
   - 至少12个字符
   - 包含大小写字母、数字和特殊字符

3. **限制数据库用户权限**
   - 只授予必要的权限
   - 不要使用root用户

4. **启用SSL连接**(生产环境)
   ```bash
   DATABASE_URL=mysql+aiomysql://user:pass@host:3306/db?ssl=true
   ```

5. **定期备份数据库**
   ```bash
   mysqldump -u watermark_user -p watermark_system > backup.sql
   ```

## 相关文件

- `shared/config.py` - 数据库配置类
- `storage/database_manager.py` - 数据库管理器
- `.env.example` - 配置模板
- `scripts/test_mysql_connection.py` - 连接测试脚本
- `requirements.txt` - 依赖包列表

## 参考资源

- [aiomysql文档](https://aiomysql.readthedocs.io/)
- [SQLAlchemy异步文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [MySQL官方文档](https://dev.mysql.com/doc/)
