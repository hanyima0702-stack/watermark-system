-- Migration: 001_create_users_table.sql
-- Description: Create users table with authentication fields
-- Requirements: 8.1, 8.4

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY COMMENT 'User UUID',
    username VARCHAR(100) NOT NULL COMMENT 'Unique username',
    email VARCHAR(255) NOT NULL COMMENT 'User email address',
    department VARCHAR(100) DEFAULT NULL COMMENT 'User department',
    roles JSON DEFAULT NULL COMMENT 'User roles',
    password_hash VARCHAR(255) NOT NULL COMMENT 'Bcrypt password hash',
    salt VARCHAR(255) NOT NULL COMMENT 'Password salt',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Account active status',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Account creation time',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
    INDEX idx_username (username),
    INDEX idx_email (email),
    INDEX idx_is_active (is_active),
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User accounts table';
