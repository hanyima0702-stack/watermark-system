-- Migration: 002_create_files_table.sql
-- Description: Create files table for storing file metadata
-- Requirements: 8.2, 8.5

CREATE TABLE IF NOT EXISTS files (
    id CHAR(36) PRIMARY KEY COMMENT 'File UUID',
    user_id CHAR(36) NOT NULL COMMENT 'Owner user ID',
    filename VARCHAR(255) NOT NULL COMMENT 'Stored filename',
    original_filename VARCHAR(255) NOT NULL COMMENT 'Original uploaded filename',
    file_size BIGINT NOT NULL COMMENT 'File size in bytes',
    file_type VARCHAR(50) NOT NULL COMMENT 'File type category (video, document, audio, image)',
    mime_type VARCHAR(100) NOT NULL COMMENT 'MIME type',
    storage_path VARCHAR(500) NOT NULL COMMENT 'Storage path',
    minio_bucket VARCHAR(100) NOT NULL COMMENT 'MinIO bucket name',
    minio_object_key VARCHAR(500) NOT NULL COMMENT 'MinIO object key',
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Upload timestamp',
    status VARCHAR(20) DEFAULT 'active' COMMENT 'File status (active, deleted, processing)',
    checksum VARCHAR(64) NULL COMMENT 'File checksum (SHA256)',
    INDEX idx_user_id (user_id),
    INDEX idx_upload_time (upload_time),
    INDEX idx_status (status),
    INDEX idx_file_type (file_type),
    CONSTRAINT fk_files_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='File metadata table';
