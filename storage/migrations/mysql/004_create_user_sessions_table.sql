-- Migration: 004_create_user_sessions_table.sql
-- Description: Create user_sessions table for session management
-- Requirements: 7.1

CREATE TABLE IF NOT EXISTS user_sessions (
    id CHAR(36) PRIMARY KEY COMMENT 'Session UUID',
    user_id CHAR(36) NOT NULL COMMENT 'User ID',
    token_hash VARCHAR(255) NOT NULL COMMENT 'JWT token hash',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Session creation time',
    expires_at TIMESTAMP NOT NULL COMMENT 'Session expiration time',
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last activity timestamp',
    ip_address VARCHAR(45) NULL COMMENT 'Client IP address (IPv4 or IPv6)',
    user_agent VARCHAR(500) NULL COMMENT 'Client user agent string',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Session active status',
    INDEX idx_user_id (user_id),
    INDEX idx_token_hash (token_hash),
    INDEX idx_expires_at (expires_at),
    INDEX idx_is_active (is_active),
    CONSTRAINT fk_user_sessions_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User sessions table';
