"""
暗水印系统配置模型

该模块定义了暗水印系统的配置数据类。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml


@dataclass
class WatermarkConfig:
    """水印配置"""
    
    # 编码配置
    ecc_type: str = "bch"  # bch或rs
    ecc_n: int = 127
    ecc_k: int = 64
    scramble_seed: int = 12345
    
    # 宏块配置
    block_size: int = 32
    header_pattern: str = "1110010"
    
    # PPM配置
    modulation_strength: int = 10
    color_space: str = "yuv"
    
    # 提取配置
    min_confidence: float = 0.3
    max_rotation: float = 45.0
    max_scale: float = 2.0
    enable_180_retry: bool = True
    
    # 性能配置
    enable_gpu: bool = False
    num_threads: int = 4
    enable_cache: bool = True
    
    # 可视化配置
    visualization_enabled: bool = False
    visualization_output_dir: str = "./debug_output"
    save_fft_spectrum: bool = True
    save_block_map: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WatermarkConfig':
        """
        从字典创建配置对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            WatermarkConfig实例
        """
        # 提取嵌套的配置
        encoding = config_dict.get('encoding', {})
        macro_block = config_dict.get('macro_block', {})
        modulation = config_dict.get('modulation', {})
        extraction = config_dict.get('extraction', {})
        performance = config_dict.get('performance', {})
        visualization = config_dict.get('visualization', {})
        
        return cls(
            # 编码配置
            ecc_type=encoding.get('type', 'bch'),
            ecc_n=encoding.get('n', 127),
            ecc_k=encoding.get('k', 64),
            scramble_seed=encoding.get('scramble_seed', 12345),
            
            # 宏块配置
            block_size=macro_block.get('size', 32),
            header_pattern=macro_block.get('header_pattern', '1110010'),
            
            # PPM配置
            modulation_strength=modulation.get('strength', 10),
            color_space=modulation.get('color_space', 'yuv'),
            
            # 提取配置
            min_confidence=extraction.get('min_confidence', 0.3),
            max_rotation=extraction.get('max_rotation', 45.0),
            max_scale=extraction.get('max_scale', 2.0),
            enable_180_retry=extraction.get('enable_180_retry', True),
            
            # 性能配置
            enable_gpu=performance.get('enable_gpu', False),
            num_threads=performance.get('num_threads', 4),
            enable_cache=performance.get('enable_cache', True),
            
            # 可视化配置
            visualization_enabled=visualization.get('enabled', False),
            visualization_output_dir=visualization.get('output_dir', './debug_output'),
            save_fft_spectrum=visualization.get('save_fft_spectrum', True),
            save_block_map=visualization.get('save_block_map', True)
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'WatermarkConfig':
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML配置文件路径
            
        Returns:
            WatermarkConfig实例
            
        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件格式错误
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        
        if not isinstance(config_dict, dict):
            raise ValueError("配置文件必须包含字典格式的数据")
        
        # 提取invisible_watermark部分
        invisible_config = config_dict.get('invisible_watermark', {})
        
        if not invisible_config:
            # 如果没有invisible_watermark键，尝试直接使用根配置
            invisible_config = config_dict
        
        return cls.from_dict(invisible_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            配置字典
        """
        return {
            'encoding': {
                'type': self.ecc_type,
                'n': self.ecc_n,
                'k': self.ecc_k,
                'scramble_seed': self.scramble_seed
            },
            'macro_block': {
                'size': self.block_size,
                'header_pattern': self.header_pattern
            },
            'modulation': {
                'strength': self.modulation_strength,
                'color_space': self.color_space
            },
            'extraction': {
                'min_confidence': self.min_confidence,
                'max_rotation': self.max_rotation,
                'max_scale': self.max_scale,
                'enable_180_retry': self.enable_180_retry
            },
            'performance': {
                'enable_gpu': self.enable_gpu,
                'num_threads': self.num_threads,
                'enable_cache': self.enable_cache
            },
            'visualization': {
                'enabled': self.visualization_enabled,
                'output_dir': self.visualization_output_dir,
                'save_fft_spectrum': self.save_fft_spectrum,
                'save_block_map': self.save_block_map
            }
        }
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        保存配置到YAML文件
        
        Args:
            yaml_path: YAML配置文件路径
        """
        config_dict = {
            'invisible_watermark': self.to_dict()
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
            
        Raises:
            ValueError: 配置无效时抛出
        """
        if self.ecc_type not in ['bch', 'rs']:
            raise ValueError(f"不支持的编码类型: {self.ecc_type}")
        
        if self.block_size not in [16, 32, 64]:
            raise ValueError(f"不支持的宏块大小: {self.block_size}")
        
        if self.modulation_strength < 1 or self.modulation_strength > 50:
            raise ValueError(f"调制强度必须在1-50之间: {self.modulation_strength}")
        
        if self.min_confidence < 0 or self.min_confidence > 1:
            raise ValueError(f"最小置信度必须在0-1之间: {self.min_confidence}")
        
        if len(self.header_pattern) < 4:
            raise ValueError(f"Header模式长度必须至少为4位: {self.header_pattern}")
        
        return True


@dataclass
class EmbedResult:
    """嵌入结果"""
    success: bool
    watermark_data: str  # 64位原始数据
    encoded_data: str  # 128位编码数据
    block_count: tuple  # (行数, 列数)
    processing_time: float
    image_size: tuple  # (height, width)
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # PSNR, SSIM等
    error_message: Optional[str] = None


@dataclass
class ExtractionResult:
    """提取结果"""
    success: bool
    watermark_data: Optional[str] = None  # 64位提取数据
    confidence: float = 0.0  # 整体置信度
    bit_confidences: Optional[list] = None  # 每位的置信度
    
    # 几何参数
    detected_rotation: float = 0.0
    detected_scale: float = 1.0
    grid_offset: tuple = (0, 0)
    
    # 统计信息
    total_blocks: int = 0
    valid_blocks: int = 0
    error_rate: float = 0.0
    
    # 性能指标
    processing_time: float = 0.0
    
    # 可视化数据
    fft_spectrum: Optional[Any] = None
    block_map: Optional[Any] = None
    
    error_message: Optional[str] = None
