-- PostgreSQL 数据库初始化脚本
-- 创建水印系统的核心数据表

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    department VARCHAR(100),
    roles TEXT[] DEFAULT '{}',
    password_hash VARCHAR(255),
    salt VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 文件元数据表
CREATE TABLE IF NOT EXISTS file_metadata (
    file_id VARCHAR(50) PRIMARY KEY,
    original_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size BIGINT NOT NULL,
    storage_path VARCHAR(500) NOT NULL,
    uploaded_by VARCHAR(50) NOT NULL REFERENCES users(user_id),
    metadata JSONB DEFAULT '{}',
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_file_hash (file_hash),
    INDEX idx_uploaded_by (uploaded_by),
    INDEX idx_uploaded_at (uploaded_at)
);

-- 水印配置表
CREATE TABLE IF NOT EXISTS watermark_configs (
    config_id VARCHAR(50) PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL,
    watermark_type VARCHAR(20) NOT NULL CHECK (watermark_type IN ('visible', 'invisible', 'both')),
    visible_config JSONB DEFAULT '{}',
    invisible_config JSONB DEFAULT '{}',
    template_variables JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(50) NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_config_name (config_name),
    INDEX idx_created_by (created_by),
    INDEX idx_is_active (is_active)
);

-- 水印任务表
CREATE TABLE IF NOT EXISTS watermark_tasks (
    task_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    file_id VARCHAR(50) NOT NULL REFERENCES file_metadata(file_id),
    config_id VARCHAR(50) NOT NULL REFERENCES watermark_configs(config_id),
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    progress DECIMAL(5,2) DEFAULT 0.00,
    error_message TEXT,
    processing_engine VARCHAR(50),
    output_file_id VARCHAR(50) REFERENCES file_metadata(file_id),
    processing_time DECIMAL(10,3),
    quality_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- 索引
    INDEX idx_user_id (user_id),
    INDEX idx_file_id (file_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_task_type (task_type)
);

-- 水印提取结果表
CREATE TABLE IF NOT EXISTS extraction_results (
    result_id VARCHAR(50) PRIMARY KEY,
    file_id VARCHAR(50) NOT NULL REFERENCES file_metadata(file_id),
    extracted_user_id VARCHAR(50) REFERENCES users(user_id),
    extracted_timestamp TIMESTAMP WITH TIME ZONE,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    extraction_method VARCHAR(50) NOT NULL,
    extraction_details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_file_id_extraction (file_id),
    INDEX idx_extracted_user_id (extracted_user_id),
    INDEX idx_confidence_score (confidence_score),
    INDEX idx_extraction_method (extraction_method)
);

-- 证据报告表
CREATE TABLE IF NOT EXISTS evidence_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    result_id VARCHAR(50) NOT NULL REFERENCES extraction_results(result_id),
    report_path VARCHAR(500) NOT NULL,
    report_metadata JSONB DEFAULT '{}',
    generated_by VARCHAR(50) NOT NULL REFERENCES users(user_id),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_result_id (result_id),
    INDEX idx_generated_by (generated_by),
    INDEX idx_generated_at (generated_at)
);

-- 审计日志表 (基础表，详细日志存储在ClickHouse)
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL REFERENCES users(user_id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    details JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    INDEX idx_user_id_audit (user_id),
    INDEX idx_action (action),
    INDEX idx_resource_type (resource_type),
    INDEX idx_timestamp_audit (timestamp)
);

-- 系统配置表
CREATE TABLE IF NOT EXISTS system_configs (
    config_key VARCHAR(100) PRIMARY KEY,
    config_value JSONB NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT false,
    updated_by VARCHAR(50) REFERENCES users(user_id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 密钥管理表
CREATE TABLE IF NOT EXISTS key_management (
    key_id VARCHAR(50) PRIMARY KEY,
    key_type VARCHAR(50) NOT NULL CHECK (key_type IN ('encryption', 'signing', 'watermark')),
    key_data TEXT NOT NULL, -- 加密存储
    key_version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(50) NOT NULL REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    rotated_at TIMESTAMP WITH TIME ZONE,
    
    -- 索引
    INDEX idx_key_type (key_type),
    INDEX idx_is_active_key (is_active),
    INDEX idx_expires_at (expires_at)
);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表添加更新时间触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watermark_configs_updated_at BEFORE UPDATE ON watermark_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watermark_tasks_updated_at BEFORE UPDATE ON watermark_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_configs_updated_at BEFORE UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认系统配置
INSERT INTO system_configs (config_key, config_value, description) VALUES
('max_file_size_mb', '500', '最大文件大小限制(MB)'),
('max_batch_files', '100', '批量处理最大文件数'),
('task_timeout_seconds', '3600', '任务超时时间(秒)'),
('cache_expire_seconds', '3600', '缓存过期时间(秒)'),
('supported_document_formats', '["docx", "xlsx", "pptx", "pdf"]', '支持的文档格式'),
('supported_image_formats', '["jpg", "jpeg", "png", "bmp", "tiff"]', '支持的图像格式'),
('supported_video_formats', '["mp4", "avi", "mov", "mkv", "wmv"]', '支持的视频格式'),
('supported_audio_formats', '["mp3", "wav", "flac", "aac", "ogg"]', '支持的音频格式')
ON CONFLICT (config_key) DO NOTHING;

-- 创建默认管理员用户 (密码: admin123)
INSERT INTO users (user_id, username, email, department, roles, password_hash, salt, is_active) VALUES
('admin', 'administrator', 'admin@watermark-system.com', 'IT', ARRAY['admin', 'operator'], 
 '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 'default_salt', true)
ON CONFLICT (user_id) DO NOTHING;

-- 创建视图：任务统计
CREATE OR REPLACE VIEW task_statistics AS
SELECT 
    user_id,
    status,
    COUNT(*) as task_count,
    AVG(processing_time) as avg_processing_time,
    DATE_TRUNC('day', created_at) as date
FROM watermark_tasks
GROUP BY user_id, status, DATE_TRUNC('day', created_at);

-- 创建视图：文件类型统计
CREATE OR REPLACE VIEW file_type_statistics AS
SELECT 
    file_type,
    COUNT(*) as file_count,
    SUM(file_size) as total_size,
    AVG(file_size) as avg_size
FROM file_metadata
GROUP BY file_type;

-- 创建索引优化查询性能
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_watermark_tasks_composite 
ON watermark_tasks (user_id, status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_file_metadata_composite 
ON file_metadata (uploaded_by, file_type, uploaded_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_extraction_results_composite 
ON extraction_results (file_id, confidence_score DESC, created_at DESC);

-- 创建分区表（用于大量审计日志）
-- 按月分区审计日志表
CREATE TABLE IF NOT EXISTS audit_logs_partitioned (
    LIKE audit_logs INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- 创建当前月份的分区
CREATE TABLE IF NOT EXISTS audit_logs_current PARTITION OF audit_logs_partitioned
FOR VALUES FROM (DATE_TRUNC('month', CURRENT_DATE)) 
TO (DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month');

COMMIT;