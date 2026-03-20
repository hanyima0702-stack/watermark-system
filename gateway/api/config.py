"""
API Gateway Configuration
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """API网关配置"""
    
    # 基础配置
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS配置
    CORS_ORIGINS: List[str] = ["*"]
    
    # 限流配置
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100  # 每分钟请求数
    RATE_LIMIT_WINDOW: int = 60  # 时间窗口（秒）
    
    # Redis配置（用于限流和缓存）
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # 认证配置
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    
    # 服务发现配置
    BUSINESS_SERVICE_URL: str = "http://localhost:8001"
    STORAGE_SERVICE_URL: str = "http://localhost:8002"
    ENGINE_SERVICE_URL: str = "http://localhost:8003"
    
    # 负载均衡配置
    LOAD_BALANCER_STRATEGY: str = "round_robin"  # round_robin, least_connections, random
    
    # 超时配置
    REQUEST_TIMEOUT: int = 300  # 5分钟
    UPLOAD_TIMEOUT: int = 600  # 10分钟
    
    # 文件上传配置
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB
    ALLOWED_EXTENSIONS: List[str] = [
        ".docx", ".xlsx", ".pptx", ".pdf",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff",
        ".mp4", ".avi", ".mov", ".mkv",
        ".mp3", ".wav", ".flac"
    ]
    
    # 监控配置
    METRICS_ENABLED: bool = True
    LOGGING_LEVEL: str = "INFO"
    
    # API版本兼容性
    SUPPORTED_API_VERSIONS: List[str] = ["v1", "v2"]
    DEFAULT_API_VERSION: str = "v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
