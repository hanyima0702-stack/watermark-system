-- ClickHouse 审计日志数据库初始化脚本
-- 用于存储大量的审计日志和时序数据

-- 创建数据库
CREATE DATABASE IF NOT EXISTS watermark_audit;

-- 使用数据库
USE watermark_audit;

-- 详细审计日志表 (按日期分区)
CREATE TABLE IF NOT EXISTS audit_logs_detailed (
    log_id String,
    user_id String,
    username String,
    action String,
    resource_type String,
    resource_id String,
    ip_address String,
    user_agent String,
    request_id String,
    session_id String,
    details String, -- JSON格式的详细信息
    success UInt8, -- 0: 失败, 1: 成功
    error_code String,
    error_message String,
    processing_time Float64, -- 处理时间(秒)
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id, timestamp)
TTL date + INTERVAL 2 YEAR; -- 数据保留2年

-- 系统性能指标表
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_name String,
    metric_value Float64,
    labels String, -- JSON格式的标签
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, metric_name, timestamp)
TTL date + INTERVAL 6 MONTH; -- 性能数据保留6个月

-- 任务执行日志表
CREATE TABLE IF NOT EXISTS task_execution_logs (
    task_id String,
    user_id String,
    file_id String,
    task_type String,
    engine_type String,
    status String,
    progress Float32,
    start_time DateTime64(3, 'UTC'),
    end_time DateTime64(3, 'UTC'),
    processing_time Float64,
    file_size UInt64,
    input_format String,
    output_format String,
    watermark_config String, -- JSON格式
    quality_metrics String, -- JSON格式
    error_details String,
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id, task_type, timestamp)
TTL date + INTERVAL 1 YEAR;

-- 文件访问日志表
CREATE TABLE IF NOT EXISTS file_access_logs (
    file_id String,
    user_id String,
    access_type String, -- upload, download, view, delete
    file_name String,
    file_type String,
    file_size UInt64,
    ip_address String,
    user_agent String,
    success UInt8,
    error_message String,
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, file_id, user_id, timestamp)
TTL date + INTERVAL 1 YEAR;

-- 水印提取日志表
CREATE TABLE IF NOT EXISTS watermark_extraction_logs (
    extraction_id String,
    file_id String,
    original_file_name String,
    extraction_method String,
    extracted_user_id String,
    extracted_timestamp DateTime64(3, 'UTC'),
    confidence_score Float32,
    processing_time Float64,
    file_size UInt64,
    success UInt8,
    error_details String,
    algorithm_params String, -- JSON格式
    quality_assessment String, -- JSON格式
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, file_id, extraction_method, timestamp)
TTL date + INTERVAL 2 YEAR;

-- 安全事件日志表
CREATE TABLE IF NOT EXISTS security_events (
    event_id String,
    event_type String, -- login_failed, unauthorized_access, suspicious_activity
    user_id String,
    ip_address String,
    user_agent String,
    resource_accessed String,
    severity String, -- low, medium, high, critical
    details String, -- JSON格式
    timestamp DateTime64(3, 'UTC'),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, severity, event_type, timestamp)
TTL date + INTERVAL 3 YEAR; -- 安全日志保留3年

-- 创建物化视图用于实时统计

-- 每小时任务统计视图
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_task_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, hour, task_type, status)
AS SELECT
    toDate(timestamp) as date,
    toHour(timestamp) as hour,
    task_type,
    status,
    count() as task_count,
    sum(processing_time) as total_processing_time,
    avg(processing_time) as avg_processing_time,
    sum(file_size) as total_file_size
FROM task_execution_logs
GROUP BY date, hour, task_type, status;

-- 每日用户活动统计视图
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_user_activity
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id)
AS SELECT
    toDate(timestamp) as date,
    user_id,
    count() as action_count,
    uniq(action) as unique_actions,
    countIf(success = 1) as success_count,
    countIf(success = 0) as failure_count
FROM audit_logs_detailed
GROUP BY date, user_id;

-- 每日文件类型统计视图
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_file_type_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, file_type, access_type)
AS SELECT
    toDate(timestamp) as date,
    file_type,
    access_type,
    count() as access_count,
    sum(file_size) as total_size,
    uniq(user_id) as unique_users
FROM file_access_logs
GROUP BY date, file_type, access_type;

-- 水印提取成功率统计视图
CREATE MATERIALIZED VIEW IF NOT EXISTS extraction_success_rate
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, extraction_method)
AS SELECT
    toDate(timestamp) as date,
    extraction_method,
    count() as total_extractions,
    countIf(success = 1) as successful_extractions,
    avg(confidence_score) as avg_confidence,
    avg(processing_time) as avg_processing_time
FROM watermark_extraction_logs
GROUP BY date, extraction_method;

-- 创建字典表用于快速查找

-- 用户信息字典
CREATE DICTIONARY IF NOT EXISTS user_dict (
    user_id String,
    username String,
    department String,
    roles Array(String)
)
PRIMARY KEY user_id
SOURCE(POSTGRESQL(
    host 'postgres'
    port 5432
    user 'watermark_user'
    password 'watermark_pass'
    db 'watermark_system'
    table 'users'
))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600); -- 5-10分钟更新一次

-- 文件类型映射字典
CREATE DICTIONARY IF NOT EXISTS file_type_dict (
    extension String,
    category String,
    mime_type String
)
PRIMARY KEY extension
SOURCE(CLICKHOUSE(
    host 'localhost'
    port 9000
    user 'default'
    password ''
    db 'watermark_audit'
    table 'file_type_mapping'
))
LAYOUT(FLAT())
LIFETIME(3600); -- 1小时更新一次

-- 创建文件类型映射表
CREATE TABLE IF NOT EXISTS file_type_mapping (
    extension String,
    category String,
    mime_type String
) ENGINE = Memory;

-- 插入文件类型映射数据
INSERT INTO file_type_mapping VALUES
('docx', 'document', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
('xlsx', 'document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
('pptx', 'document', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'),
('pdf', 'document', 'application/pdf'),
('jpg', 'image', 'image/jpeg'),
('jpeg', 'image', 'image/jpeg'),
('png', 'image', 'image/png'),
('bmp', 'image', 'image/bmp'),
('tiff', 'image', 'image/tiff'),
('mp4', 'video', 'video/mp4'),
('avi', 'video', 'video/x-msvideo'),
('mov', 'video', 'video/quicktime'),
('mkv', 'video', 'video/x-matroska'),
('wmv', 'video', 'video/x-ms-wmv'),
('mp3', 'audio', 'audio/mpeg'),
('wav', 'audio', 'audio/wav'),
('flac', 'audio', 'audio/flac'),
('aac', 'audio', 'audio/aac'),
('ogg', 'audio', 'audio/ogg');

-- 创建常用查询的预聚合表

-- 每日系统概览表
CREATE TABLE IF NOT EXISTS daily_system_overview (
    date Date,
    total_tasks UInt32,
    completed_tasks UInt32,
    failed_tasks UInt32,
    total_files_processed UInt64,
    total_data_processed UInt64,
    unique_users UInt32,
    avg_processing_time Float64,
    system_load Float64
) ENGINE = MergeTree()
ORDER BY date;

-- 创建定时任务来更新概览表
-- 注意：这需要在ClickHouse配置中启用定时任务功能