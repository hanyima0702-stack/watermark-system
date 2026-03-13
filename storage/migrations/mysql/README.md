# MySQL数据库迁移脚本

本目录包含MySQL数据库的迁移脚本，用于创建用户认证和存储系统所需的数据库表结构。

## 迁移文件列表

1. **001_create_users_table.sql** - 创建用户表
   - 存储用户账号信息
   - 包含用户名、密码哈希、邮箱等字段
   - 需求: 8.1, 8.4

2. **002_create_files_table.sql** - 创建文件元数据表
   - 存储上传文件的元数据
   - 关联MinIO对象存储
   - 需求: 8.2, 8.5

3. **003_create_watermark_tasks_table.sql** - 创建水印任务表
   - 跟踪水印嵌入和提取任务
   - 关联用户和文件
   - 需求: 8.3, 8.6

4. **004_create_user_sessions_table.sql** - 创建用户会话表
   - 管理用户登录会话
   - 存储JWT令牌信息
   - 需求: 7.1

## 使用方法

### 自动执行所有迁移

使用初始化脚本自动执行所有迁移：

```bash
cd watermark-system
python scripts/init_mysql.py
```

### 手动执行迁移

如果需要手动执行迁移，可以使用MySQL客户端：

```bash
mysql -h localhost -u watermark_user -p watermark_system < storage/migrations/mysql/001_create_users_table.sql
mysql -h localhost -u watermark_user -p watermark_system < storage/migrations/mysql/002_create_files_table.sql
mysql -h localhost -u watermark_user -p watermark_system < storage/migrations/mysql/003_create_watermark_tasks_table.sql
mysql -h localhost -u watermark_user -p watermark_system < storage/migrations/mysql/004_create_user_sessions_table.sql
```

## 数据库配置

确保在 `.env` 文件中配置了正确的MySQL连接信息：

```env
DATABASE_DB_TYPE=mysql
DATABASE_HOST=localhost
DATABASE_PORT=3306
DATABASE_USER=watermark_user
DATABASE_PASSWORD=watermark_pass
DATABASE_DATABASE=watermark_system
DATABASE_CHARSET=utf8mb4
```

## 表结构说明

### users 表
- 主键: id (UUID)
- 唯一索引: username, email
- 字符集: utf8mb4

### files 表
- 主键: id (UUID)
- 外键: user_id -> users(id)
- 索引: user_id, upload_time, status

### watermark_tasks 表
- 主键: id (UUID)
- 外键: user_id -> users(id), file_id -> files(id)
- 索引: user_id, file_id, status, created_at

### user_sessions 表
- 主键: id (UUID)
- 外键: user_id -> users(id)
- 索引: user_id, token_hash, expires_at

## 注意事项

1. 所有表使用 InnoDB 引擎，支持事务和外键约束
2. 字符集统一使用 utf8mb4，支持完整的Unicode字符
3. 外键约束使用 CASCADE 删除，确保数据一致性
4. 时间戳字段使用 TIMESTAMP 类型，自动管理创建和更新时间
5. 所有表都包含适当的索引以优化查询性能

## 回滚

如果需要删除所有表，可以按相反顺序执行：

```sql
DROP TABLE IF EXISTS user_sessions;
DROP TABLE IF EXISTS watermark_tasks;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS users;
```

或使用脚本：

```bash
mysql -h localhost -u watermark_user -p watermark_system -e "
DROP TABLE IF EXISTS user_sessions;
DROP TABLE IF EXISTS watermark_tasks;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS users;
"
```
