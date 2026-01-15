"""
系统配置管理
统一管理所有服务的配置参数
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator
from functools import lru_cache


class DatabaseConfig(BaseSettings):
    """数据库配置"""
    url: str = "postgresql://watermark_user:watermark_pass@localhost:5432/watermark_system"
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False
    
    class Config:
        env_prefix = "DATABASE_"


class RedisConfig(BaseSettings):
    """Redis配置"""
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    
    class Config:
        env_prefix = "REDIS_"


class ClickHouseConfig(BaseSettings):
    """ClickHouse配置"""
    host: str = "localhost"
    port: int = 8123
    user: str = "default"
    password: str = ""
    database: str = "watermark_audit"
    
    class Config:
        env_prefix = "CLICKHOUSE_"


class MessageQueueConfig(BaseSettings):
    """消息队列配置"""
    rabbitmq_url: str = "amqp://watermark:watermark123@localhost:5672/"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    
    class Config:
        env_prefix = "MQ_"


class ObjectStorageConfig(BaseSettings):
    """对象存储配置"""
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin123"
    bucket_name: str = "watermark-files"
    secure: bool = False
    region: str = "us-east-1"
    
    class Config:
        env_prefix = "MINIO_"


class SecurityConfig(BaseSettings):
    """安全配置"""
    secret_key: str = "your-secret-key-here-change-in-production"
    jwt_secret_key: str = "your-jwt-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24小时
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    class Config:
        env_prefix = "SECURITY_"


class LDAPConfig(BaseSettings):
    """LDAP/SSO配置"""
    server: Optional[str] = None
    base_dn: Optional[str] = None
    bind_dn: Optional[str] = None
    bind_password: Optional[str] = None
    user_search_base: Optional[str] = None
    group_search_base: Optional[str] = None
    enabled: bool = False
    
    class Config:
        env_prefix = "LDAP_"


class FileProcessingConfig(BaseSettings):
    """文件处理配置"""
    max_file_size_mb: int = 500
    max_batch_files: int = 100
    temp_dir: str = "/tmp/watermark"
    upload_dir: str = "/data/uploads"
    output_dir: str = "/data/outputs"
    supported_document_formats: List[str] = [".docx", ".xlsx", ".pptx", ".pdf"]
    supported_image_formats: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    supported_video_formats: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
    supported_audio_formats: List[str] = [".mp3", ".wav", ".flac", ".aac", ".ogg"]
    
    @validator('temp_dir', 'upload_dir', 'output_dir')
    def create_directories(cls, v):
        os.makedirs(v, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "FILE_"


class WatermarkConfig(BaseSettings):
    """水印配置"""
    default_opacity: float = 0.5
    default_font_size: int = 12
    default_color: str = "#FF0000"
    key_id: str = "default-watermark-key"
    max_layers: int = 10
    
    class Config:
        env_prefix = "WATERMARK_"


class PerformanceConfig(BaseSettings):
    """性能配置"""
    worker_processes: int = 4
    worker_threads: int = 2
    task_timeout_seconds: int = 3600
    cache_expire_seconds: int = 3600
    max_concurrent_tasks: int = 100
    gpu_enabled: bool = False
    
    class Config:
        env_prefix = "PERFORMANCE_"


class MonitoringConfig(BaseSettings):
    """监控配置"""
    prometheus_port: int = 9090
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    
    class Config:
        env_prefix = "MONITORING_"


class APIConfig(BaseSettings):
    """API配置"""
    v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000"]
    rate_limit_per_minute: int = 100
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_prefix = "API_"


class EmailConfig(BaseSettings):
    """邮件配置"""
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_tls: bool = True
    from_email: Optional[str] = None
    enabled: bool = False
    
    class Config:
        env_prefix = "EMAIL_"


class BackupConfig(BaseSettings):
    """备份配置"""
    enabled: bool = True
    schedule: str = "0 2 * * *"  # 每天凌晨2点
    retention_days: int = 30
    storage_path: str = "/data/backups"
    
    class Config:
        env_prefix = "BACKUP_"


class AppConfig(BaseSettings):
    """应用主配置"""
    debug: bool = False
    testing: bool = False
    environment: str = "production"
    app_name: str = "Watermark System"
    app_version: str = "1.0.0"
    
    # 子配置
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    clickhouse: ClickHouseConfig = ClickHouseConfig()
    message_queue: MessageQueueConfig = MessageQueueConfig()
    object_storage: ObjectStorageConfig = ObjectStorageConfig()
    security: SecurityConfig = SecurityConfig()
    ldap: LDAPConfig = LDAPConfig()
    file_processing: FileProcessingConfig = FileProcessingConfig()
    watermark: WatermarkConfig = WatermarkConfig()
    performance: PerformanceConfig = PerformanceConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()
    email: EmailConfig = EmailConfig()
    backup: BackupConfig = BackupConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> AppConfig:
    """获取应用配置单例"""
    return AppConfig()


# 配置验证函数
def validate_config(config: AppConfig) -> List[str]:
    """验证配置的有效性"""
    errors = []
    
    # 检查必需的配置
    if config.environment == "production":
        if config.security.secret_key == "your-secret-key-here-change-in-production":
            errors.append("生产环境必须设置 SECRET_KEY")
        
        if config.security.jwt_secret_key == "your-jwt-secret-key-here":
            errors.append("生产环境必须设置 JWT_SECRET_KEY")
    
    # 检查文件路径
    required_dirs = [
        config.file_processing.temp_dir,
        config.file_processing.upload_dir,
        config.file_processing.output_dir
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"无法创建目录 {dir_path}: {e}")
    
    # 检查数值范围
    if config.file_processing.max_file_size_mb <= 0:
        errors.append("max_file_size_mb 必须大于 0")
    
    if config.performance.worker_processes <= 0:
        errors.append("worker_processes 必须大于 0")
    
    return errors


# 环境特定配置
class DevelopmentConfig(AppConfig):
    """开发环境配置"""
    debug: bool = True
    environment: str = "development"
    
    class Config:
        env_file = ".env.dev"


class TestingConfig(AppConfig):
    """测试环境配置"""
    testing: bool = True
    environment: str = "testing"
    
    class Config:
        env_file = ".env.test"


class ProductionConfig(AppConfig):
    """生产环境配置"""
    debug: bool = False
    environment: str = "production"
    
    class Config:
        env_file = ".env.prod"


# 配置工厂函数
def get_config_by_env(env: str = None) -> AppConfig:
    """根据环境获取配置"""
    env = env or os.getenv("ENVIRONMENT", "production")
    
    config_map = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig
    }
    
    config_class = config_map.get(env, ProductionConfig)
    return config_class()


# 导出常用配置
settings = get_settings()