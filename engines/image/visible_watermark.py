"""
明水印处理器 - 实现可见水印的生成和嵌入
支持文字水印、二维码水印、多层水印叠加等功能
"""

import io
import math
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import qrcode
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum


class WatermarkType(Enum):
    """水印类型枚举"""
    TEXT = "text"
    QRCODE = "qrcode"
    BARCODE = "barcode"
    IMAGE = "image"


class PositionType(Enum):
    """位置类型枚举"""
    CENTER = "center"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    CUSTOM = "custom"


@dataclass
class FontConfig:
    """字体配置"""
    family: str = "Arial"
    size: int = 24
    color: str = "#FF0000"
    opacity: float = 0.5
    bold: bool = False
    italic: bool = False


@dataclass
class PositionConfig:
    """位置配置"""
    type: PositionType = PositionType.CENTER
    x: Union[int, str] = "center"
    y: Union[int, str] = "center"
    rotation: float = 0.0
    spacing_x: int = 100
    spacing_y: int = 100


@dataclass
class WatermarkLayer:
    """水印层配置"""
    type: WatermarkType
    content: str
    font_config: Optional[FontConfig] = None
    position_config: Optional[PositionConfig] = None
    size: Optional[int] = None
    opacity: float = 0.5
    enabled: bool = True


class VisibleWatermarkProcessor:
    """可见水印处理器"""
    
    def __init__(self):
        self.default_font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
    
    def generate_text_watermark(
        self, 
        text: str, 
        font_config: FontConfig,
        canvas_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        生成文字水印
        
        Args:
            text: 水印文字内容
            font_config: 字体配置
            canvas_size: 画布大小，如果为None则自动计算
            
        Returns:
            PIL Image对象
        """
        # 加载字体
        font = self._load_font(font_config.family, font_config.size)
        
        # 计算文字尺寸
        if canvas_size is None:
            # 创建临时图像来测量文字尺寸
            temp_img = Image.new('RGBA', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            canvas_size = (text_width + 20, text_height + 20)
        
        # 创建透明画布
        watermark = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # 解析颜色
        color = self._parse_color(font_config.color, font_config.opacity)
        
        # 计算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (canvas_size[0] - text_width) // 2
        y = (canvas_size[1] - text_height) // 2
        
        # 绘制文字
        draw.text((x, y), text, font=font, fill=color)
        
        return watermark
    
    def generate_qr_watermark(
        self, 
        data: str, 
        size: int = 100,
        error_correction: int = qrcode.constants.ERROR_CORRECT_M
    ) -> Image.Image:
        """
        生成二维码水印
        
        Args:
            data: 二维码数据
            size: 二维码尺寸
            error_correction: 错误纠正级别
            
        Returns:
            PIL Image对象
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=error_correction,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # 生成二维码图像
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # 调整尺寸
        qr_img = qr_img.resize((size, size), Image.Resampling.LANCZOS)
        
        # 转换为RGBA模式以支持透明度
        if qr_img.mode != 'RGBA':
            qr_img = qr_img.convert('RGBA')
        
        return qr_img
    
    def generate_barcode_watermark(
        self, 
        data: str, 
        width: int = 200, 
        height: int = 50
    ) -> Image.Image:
        """
        生成条形码水印（简单实现）
        
        Args:
            data: 条形码数据
            width: 条形码宽度
            height: 条形码高度
            
        Returns:
            PIL Image对象
        """
        # 简单的条形码实现（实际项目中可以使用python-barcode库）
        barcode = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(barcode)
        
        # 将数据转换为二进制表示
        binary_data = ''.join(format(ord(c), '08b') for c in data)
        
        # 绘制条形码
        bar_width = width // len(binary_data)
        for i, bit in enumerate(binary_data):
            if bit == '1':
                x1 = i * bar_width
                x2 = (i + 1) * bar_width
                draw.rectangle([x1, 0, x2, height], fill=(0, 0, 0, 255))
        
        return barcode
    
    def apply_watermark(
        self, 
        image: Image.Image, 
        watermark: Image.Image,
        position_config: PositionConfig,
        tile: bool = False
    ) -> Image.Image:
        """
        将水印应用到图像上
        
        Args:
            image: 原始图像
            watermark: 水印图像
            position_config: 位置配置
            tile: 是否平铺水印
            
        Returns:
            添加水印后的图像
        """
        # 确保图像为RGBA模式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        if watermark.mode != 'RGBA':
            watermark = watermark.convert('RGBA')
        
        # 旋转水印
        if position_config.rotation != 0:
            watermark = watermark.rotate(
                position_config.rotation, 
                expand=True, 
                fillcolor=(0, 0, 0, 0)
            )
        
        if tile:
            return self._apply_tiled_watermark(image, watermark, position_config)
        else:
            return self._apply_single_watermark(image, watermark, position_config)
    
    def apply_multiple_watermarks(
        self, 
        image: Image.Image, 
        watermark_layers: List[WatermarkLayer]
    ) -> Image.Image:
        """
        应用多层水印
        
        Args:
            image: 原始图像
            watermark_layers: 水印层列表
            
        Returns:
            添加多层水印后的图像
        """
        result = image.copy()
        
        for layer in watermark_layers:
            if not layer.enabled:
                continue
                
            # 生成水印
            if layer.type == WatermarkType.TEXT:
                watermark = self.generate_text_watermark(
                    layer.content, 
                    layer.font_config or FontConfig()
                )
            elif layer.type == WatermarkType.QRCODE:
                watermark = self.generate_qr_watermark(
                    layer.content, 
                    layer.size or 100
                )
            elif layer.type == WatermarkType.BARCODE:
                watermark = self.generate_barcode_watermark(layer.content)
            else:
                continue
            
            # 调整透明度
            if layer.opacity < 1.0:
                watermark = self._adjust_opacity(watermark, layer.opacity)
            
            # 应用水印
            result = self.apply_watermark(
                result, 
                watermark, 
                layer.position_config or PositionConfig()
            )
        
        return result
    
    def _load_font(self, font_family: str, font_size: int) -> ImageFont.ImageFont:
        """加载字体"""
        try:
            # 尝试加载系统字体
            for font_path in self.default_font_paths:
                try:
                    return ImageFont.truetype(font_path, font_size)
                except (OSError, IOError):
                    continue
            
            # 如果系统字体加载失败，使用默认字体
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
    
    def _parse_color(self, color_str: str, opacity: float) -> Tuple[int, int, int, int]:
        """解析颜色字符串"""
        if color_str.startswith('#'):
            # 十六进制颜色
            color_str = color_str[1:]
            if len(color_str) == 6:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
            else:
                r = g = b = 0
        else:
            # 默认为黑色
            r = g = b = 0
        
        a = int(255 * opacity)
        return (r, g, b, a)
    
    def _calculate_position(
        self, 
        image_size: Tuple[int, int], 
        watermark_size: Tuple[int, int],
        position_config: PositionConfig
    ) -> Tuple[int, int]:
        """计算水印位置"""
        img_width, img_height = image_size
        wm_width, wm_height = watermark_size
        
        if position_config.type == PositionType.CENTER:
            x = (img_width - wm_width) // 2
            y = (img_height - wm_height) // 2
        elif position_config.type == PositionType.TOP_LEFT:
            x = y = 10
        elif position_config.type == PositionType.TOP_RIGHT:
            x = img_width - wm_width - 10
            y = 10
        elif position_config.type == PositionType.BOTTOM_LEFT:
            x = 10
            y = img_height - wm_height - 10
        elif position_config.type == PositionType.BOTTOM_RIGHT:
            x = img_width - wm_width - 10
            y = img_height - wm_height - 10
        else:  # CUSTOM
            x = position_config.x if isinstance(position_config.x, int) else 0
            y = position_config.y if isinstance(position_config.y, int) else 0
        
        return (x, y)
    
    def _apply_single_watermark(
        self, 
        image: Image.Image, 
        watermark: Image.Image,
        position_config: PositionConfig
    ) -> Image.Image:
        """应用单个水印"""
        result = image.copy()
        
        # 计算位置
        x, y = self._calculate_position(
            image.size, 
            watermark.size, 
            position_config
        )
        
        # 使用alpha合成
        result.paste(watermark, (x, y), watermark)
        
        return result
    
    def _apply_tiled_watermark(
        self, 
        image: Image.Image, 
        watermark: Image.Image,
        position_config: PositionConfig
    ) -> Image.Image:
        """应用平铺水印"""
        result = image.copy()
        img_width, img_height = image.size
        wm_width, wm_height = watermark.size
        
        # 计算平铺参数
        spacing_x = position_config.spacing_x
        spacing_y = position_config.spacing_y
        
        # 平铺水印
        y = 0
        while y < img_height:
            x = 0
            while x < img_width:
                # 检查是否超出边界
                if x + wm_width <= img_width and y + wm_height <= img_height:
                    result.paste(watermark, (x, y), watermark)
                x += wm_width + spacing_x
            y += wm_height + spacing_y
        
        return result
    
    def _adjust_opacity(self, image: Image.Image, opacity: float) -> Image.Image:
        """调整图像透明度"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 创建新的alpha通道
        alpha = image.split()[-1]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        
        # 重新组合图像
        image.putalpha(alpha)
        return image


class WatermarkConfigParser:
    """水印配置解析器"""
    
    @staticmethod
    def parse_watermark_config(config: Dict) -> List[WatermarkLayer]:
        """
        解析水印配置
        
        Args:
            config: 水印配置字典
            
        Returns:
            水印层列表
        """
        layers = []
        
        if 'visible_watermark' in config and config['visible_watermark'].get('enabled', False):
            for layer_config in config['visible_watermark'].get('layers', []):
                layer = WatermarkConfigParser._parse_layer(layer_config)
                if layer:
                    layers.append(layer)
        
        return layers
    
    @staticmethod
    def _parse_layer(layer_config: Dict) -> Optional[WatermarkLayer]:
        """解析单个水印层配置"""
        try:
            # 解析水印类型
            watermark_type = WatermarkType(layer_config.get('type', 'text'))
            
            # 解析内容
            content = layer_config.get('content', '')
            
            # 解析字体配置
            font_config = None
            if 'font' in layer_config:
                font_data = layer_config['font']
                font_config = FontConfig(
                    family=font_data.get('family', 'Arial'),
                    size=font_data.get('size', 24),
                    color=font_data.get('color', '#FF0000'),
                    opacity=font_data.get('opacity', 0.5),
                    bold=font_data.get('bold', False),
                    italic=font_data.get('italic', False)
                )
            
            # 解析位置配置
            position_config = None
            if 'position' in layer_config:
                pos_data = layer_config['position']
                position_type = PositionType.CENTER
                
                # 解析位置类型
                x = pos_data.get('x', 'center')
                y = pos_data.get('y', 'center')
                
                if isinstance(x, str) and isinstance(y, str):
                    if x == 'center' and y == 'center':
                        position_type = PositionType.CENTER
                    elif x == 'left' and y == 'top':
                        position_type = PositionType.TOP_LEFT
                    elif x == 'right' and y == 'top':
                        position_type = PositionType.TOP_RIGHT
                    elif x == 'left' and y == 'bottom':
                        position_type = PositionType.BOTTOM_LEFT
                    elif x == 'right' and y == 'bottom':
                        position_type = PositionType.BOTTOM_RIGHT
                else:
                    position_type = PositionType.CUSTOM
                
                position_config = PositionConfig(
                    type=position_type,
                    x=x,
                    y=y,
                    rotation=pos_data.get('rotation', 0.0),
                    spacing_x=pos_data.get('spacing', {}).get('horizontal', 100),
                    spacing_y=pos_data.get('spacing', {}).get('vertical', 100)
                )
            
            return WatermarkLayer(
                type=watermark_type,
                content=content,
                font_config=font_config,
                position_config=position_config,
                size=layer_config.get('size'),
                opacity=layer_config.get('opacity', 0.5),
                enabled=layer_config.get('enabled', True)
            )
            
        except Exception as e:
            print(f"Error parsing watermark layer: {e}")
            return None