"""
水印嵌入模块

该模块提供宏块生成、PPM调制和图像嵌入功能。
"""

from .macro_block import MacroBlockGenerator

# PPMModulator and ImageEmbedder will be imported when implemented
try:
    from .ppm_modulator import PPMModulator
except ImportError:
    PPMModulator = None

try:
    from .image_embedder import ImageEmbedder
except ImportError:
    ImageEmbedder = None

__all__ = ['MacroBlockGenerator']
if PPMModulator is not None:
    __all__.append('PPMModulator')
if ImageEmbedder is not None:
    __all__.append('ImageEmbedder')
