-- Migration: 001_create_users_table.sql
-- Description: Create users table with authentication fields
-- Requirements: 8.1, 8.4

CREATE TABLE IF NOT EXISTS users (
    id CHAR(36) PRIMARY KEY COMMENT 'User UUID',
    username VARCHAR(50) NOT NULL COMMENT 'Unique username',
    password_hash VARCHAR(255) NOT NULL COMMENT 'Bcrypt password hash',
    salt VARCHAR(64) NOT NULL COMMENT 'Password salt',
    email VARCHAR(255) NOT NULL COMMENT 'User email address',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Account creation time',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
    last_login_at TIMESTAMP NULL COMMENT 'Last login time',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Account active status',
    INDEX idx_username (username),
    INDEX idx_email (email),
    UNIQUE KEY uk_username (username),
    UNIQUE KEY uk_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='User accounts table';
