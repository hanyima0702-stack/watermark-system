-- Migration: 003_create_watermark_tasks_table.sql
-- Description: Create watermark_tasks table for tracking watermark operations
-- Requirements: 8.3, 8.6

CREATE TABLE IF NOT EXISTS watermark_tasks (
    id CHAR(36) PRIMARY KEY COMMENT 'Task UUID',
    user_id CHAR(36) NOT NULL COMMENT 'User who created the task',
    file_id CHAR(36) NOT NULL COMMENT 'Input file ID',
    task_type VARCHAR(50) NOT NULL COMMENT 'Task type (embed, extract)',
    status VARCHAR(20) DEFAULT 'pending' COMMENT 'Task status (pending, processing, completed, failed)',
    progress INT DEFAULT 0 COMMENT 'Task progress percentage (0-100)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Task creation time',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
    completed_at TIMESTAMP NULL COMMENT 'Task completion time',
    result_file_id CHAR(36) NULL COMMENT 'Result file ID (for embed tasks)',
    error_message TEXT NULL COMMENT 'Error message if task failed',
    INDEX idx_user_id (user_id),
    INDEX idx_file_id (file_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_task_type (task_type),
    CONSTRAINT fk_watermark_tasks_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_watermark_tasks_file_id FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    CONSTRAINT fk_watermark_tasks_result_file_id FOREIGN KEY (result_file_id) REFERENCES files(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Watermark tasks table';
