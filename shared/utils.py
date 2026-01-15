"""
共享工具函数
提供系统通用的工具函数和辅助类
"""

import hashlib
import uuid
import mimetypes
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """计算文件哈希值"""
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """获取文件MIME类型"""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """获取文件扩展名"""
        return Path(filename).suffix.lower()
    
    @staticmethod
    def is_supported_format(file_extension: str, supported_formats: List[str]) -> bool:
        """检查文件格式是否支持"""
        return file_extension.lower() in [fmt.lower() for fmt in supported_formats]
    
    @staticmethod
    def generate_unique_filename(original_name: str, user_id: str) -> str:
        """生成唯一文件名"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_ext = FileUtils.get_file_extension(original_name)
        unique_id = str(uuid.uuid4())[:8]
        return f"{user_id}_{timestamp}_{unique_id}{file_ext}"


class IDGenerator:
    """ID生成器"""
    
    @staticmethod
    def generate_task_id() -> str:
        """生成任务ID"""
        return f"task_{uuid.uuid4().hex}"
    
    @staticmethod
    def generate_file_id() -> str:
        """生成文件ID"""
        return f"file_{uuid.uuid4().hex}"
    
    @staticmethod
    def generate_config_id() -> str:
        """生成配置ID"""
        return f"config_{uuid.uuid4().hex}"
    
    @staticmethod
    def generate_request_id() -> str:
        """生成请求ID"""
        return f"req_{uuid.uuid4().hex}"


class TimeUtils:
    """时间处理工具类"""
    
    @staticmethod
    def now_utc() -> datetime:
        """获取当前UTC时间"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """格式化日期时间"""
        return dt.strftime(format_str)
    
    @staticmethod
    def parse_datetime(date_str: str, format_str: str = '%Y-%m-%d %H:%M:%S') -> datetime:
        """解析日期时间字符串"""
        return datetime.strptime(date_str, format_str)


class ConfigUtils:
    """配置工具类"""
    
    @staticmethod
    def load_json_config(config_path: str) -> Dict[str, Any]:
        """加载JSON配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_json_config(config: Dict[str, Any], config_path: str) -> bool:
        """保存JSON配置文件"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Failed to save config to {config_path}: {e}")
            return False
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置字典"""
        result = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = ConfigUtils.merge_configs(result[key], value)
            else:
                result[key] = value
        return result


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """验证用户ID格式"""
        return bool(user_id and len(user_id) >= 3 and user_id.replace('_', '').replace('-', '').isalnum())
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 100) -> bool:
        """验证文件大小"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return 0 < file_size <= max_size_bytes
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清理文件名中的非法字符"""
        import re
        # 移除或替换非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(sanitized) > 255:
            name, ext = Path(sanitized).stem, Path(sanitized).suffix
            sanitized = name[:255-len(ext)] + ext
        return sanitized


class LoggerUtils:
    """日志工具类"""
    
    @staticmethod
    def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @staticmethod
    def log_request(logger: logging.Logger, request_id: str, method: str, params: Dict[str, Any]):
        """记录请求日志"""
        logger.info(f"Request {request_id}: {method} with params: {params}")
    
    @staticmethod
    def log_response(logger: logging.Logger, request_id: str, success: bool, processing_time: float):
        """记录响应日志"""
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"Response {request_id}: {status} in {processing_time:.3f}s")


class SecurityUtils:
    """安全工具类"""
    
    @staticmethod
    def generate_salt() -> str:
        """生成随机盐值"""
        return uuid.uuid4().hex
    
    @staticmethod
    def hash_password(password: str, salt: str) -> str:
        """哈希密码"""
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    @staticmethod
    def verify_password(password: str, salt: str, hashed: str) -> bool:
        """验证密码"""
        return SecurityUtils.hash_password(password, salt) == hashed
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = '*', visible_chars: int = 4) -> str:
        """掩码敏感数据"""
        if len(data) <= visible_chars:
            return mask_char * len(data)
        return data[:visible_chars] + mask_char * (len(data) - visible_chars)


class PerformanceUtils:
    """性能工具类"""
    
    @staticmethod
    def measure_time(func):
        """装饰器：测量函数执行时间"""
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        return wrapper
    
    @staticmethod
    def calculate_file_processing_speed(file_size_bytes: int, processing_time_seconds: float) -> float:
        """计算文件处理速度 (MB/s)"""
        if processing_time_seconds <= 0:
            return 0.0
        file_size_mb = file_size_bytes / (1024 * 1024)
        return file_size_mb / processing_time_seconds


class ErrorUtils:
    """错误处理工具类"""
    
    @staticmethod
    def create_error_response(error_code: str, error_message: str, request_id: str = None) -> Dict[str, Any]:
        """创建标准错误响应"""
        return {
            'success': False,
            'error_code': error_code,
            'error_message': error_message,
            'request_id': request_id,
            'timestamp': TimeUtils.now_utc().isoformat()
        }
    
    @staticmethod
    def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
        """记录错误日志"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        logger.error(f"Error occurred: {error_info}")


# 常量定义
class Constants:
    """系统常量"""
    
    # 支持的文件格式
    SUPPORTED_DOCUMENT_FORMATS = ['.docx', '.xlsx', '.pptx', '.pdf']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.aac', '.ogg']
    
    # 文件大小限制 (MB)
    MAX_FILE_SIZE_MB = 500
    MAX_BATCH_FILES = 100
    
    # 任务超时时间 (秒)
    TASK_TIMEOUT_SECONDS = 3600
    
    # 缓存过期时间 (秒)
    CACHE_EXPIRE_SECONDS = 3600
    
    # 水印配置默认值
    DEFAULT_WATERMARK_OPACITY = 0.5
    DEFAULT_WATERMARK_FONT_SIZE = 12
    DEFAULT_WATERMARK_COLOR = '#FF0000'
    
    # 错误代码
    ERROR_CODES = {
        'INVALID_FILE_FORMAT': 'E001',
        'FILE_TOO_LARGE': 'E002',
        'PROCESSING_FAILED': 'E003',
        'UNAUTHORIZED': 'E004',
        'CONFIG_NOT_FOUND': 'E005',
        'TASK_TIMEOUT': 'E006',
        'STORAGE_ERROR': 'E007',
        'EXTRACTION_FAILED': 'E008'
    }