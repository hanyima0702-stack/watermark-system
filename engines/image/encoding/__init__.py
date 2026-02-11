"""
纠错编码模块

该模块提供纠错编码和加扰功能，用于增强水印数据的容错能力。
"""

from .ecc_encoder import ECCEncoder
from .scrambler import Scrambler

__all__ = ['ECCEncoder', 'Scrambler']
