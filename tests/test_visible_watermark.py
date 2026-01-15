"""
明水印处理器单元测试
"""

import pytest
import tempfile
import os
from PIL import Image
import numpy as np

from engines.image.visible_watermark import (
    VisibleWatermarkProcessor,
    WatermarkConfigParser,
    FontConfig,
    PositionConfig,
    WatermarkLayer,
    WatermarkType,
    PositionType
)


class TestVisibleWatermarkProcessor:
    """可见水印处理器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.processor = VisibleWatermarkProcessor()
        self.test_image = Image.new('RGB', (800, 600), color='white')
    
    def test_generate_text_watermark(self):
        """测试文字水印生成"""
        font_config = FontConfig(
            family="Arial",
            size=24,
            color="#FF0000",
            opacity=0.5
        )
        
        watermark = self.processor.generate_text_watermark(
            "Test Watermark", 
            font_config
        )
        
        assert watermark is not None
        assert watermark.mode == 'RGBA'
        assert watermark.size[0] > 0
        assert watermark.size[1] > 0
    
    def test_generate_text_watermark_with_custom_size(self):
        """测试指定尺寸的文字水印生成"""
        font_config = FontConfig(size=36, color="#0000FF", opacity=0.8)
        canvas_size = (300, 100)
        
        watermark = self.processor.generate_text_watermark(
            "Custom Size", 
            font_config, 
            canvas_size
        )
        
        assert watermark.size == canvas_size
        assert watermark.mode == 'RGBA'
    
    def test_generate_qr_watermark(self):
        """测试二维码水印生成"""
        qr_data = "https://example.com"
        size = 150
        
        watermark = self.processor.generate_qr_watermark(qr_data, size)
        
        assert watermark is not None
        assert watermark.mode == 'RGBA'
        assert watermark.size == (size, size)
    
    def test_generate_barcode_watermark(self):
        """测试条形码水印生成"""
        barcode_data = "123456789"
        width, height = 200, 50
        
        watermark = self.processor.generate_barcode_watermark(
            barcode_data, 
            width, 
            height
        )
        
        assert watermark is not None
        assert watermark.mode == 'RGBA'
        assert watermark.size == (width, height)
    
    def test_apply_single_watermark_center(self):
        """测试居中应用单个水印"""
        # 生成文字水印
        font_config = FontConfig(size=20, opacity=0.6)
        watermark = self.processor.generate_text_watermark("CENTER", font_config)
        
        # 应用水印
        position_config = PositionConfig(type=PositionType.CENTER)
        result = self.processor.apply_watermark(
            self.test_image, 
            watermark, 
            position_config
        )
        
        assert result is not None
        assert result.size == self.test_image.size
        assert result.mode == 'RGBA'
    
    def test_apply_single_watermark_corners(self):
        """测试四角位置应用水印"""
        font_config = FontConfig(size=16, opacity=0.7)
        watermark = self.processor.generate_text_watermark("CORNER", font_config)
        
        positions = [
            PositionType.TOP_LEFT,
            PositionType.TOP_RIGHT,
            PositionType.BOTTOM_LEFT,
            PositionType.BOTTOM_RIGHT
        ]
        
        for pos_type in positions:
            position_config = PositionConfig(type=pos_type)
            result = self.processor.apply_watermark(
                self.test_image, 
                watermark, 
                position_config
            )
            
            assert result is not None
            assert result.size == self.test_image.size
    
    def test_apply_watermark_with_rotation(self):
        """测试旋转水印应用"""
        font_config = FontConfig(size=20)
        watermark = self.processor.generate_text_watermark("ROTATED", font_config)
        
        position_config = PositionConfig(
            type=PositionType.CENTER,
            rotation=45.0
        )
        
        result = self.processor.apply_watermark(
            self.test_image, 
            watermark, 
            position_config
        )
        
        assert result is not None
        assert result.size == self.test_image.size
    
    def test_apply_tiled_watermark(self):
        """测试平铺水印应用"""
        font_config = FontConfig(size=16, opacity=0.3)
        watermark = self.processor.generate_text_watermark("TILE", font_config)
        
        position_config = PositionConfig(
            spacing_x=50,
            spacing_y=50
        )
        
        result = self.processor.apply_watermark(
            self.test_image, 
            watermark, 
            position_config,
            tile=True
        )
        
        assert result is not None
        assert result.size == self.test_image.size
    
    def test_apply_multiple_watermarks(self):
        """测试多层水印应用"""
        layers = [
            WatermarkLayer(
                type=WatermarkType.TEXT,
                content="Layer 1",
                font_config=FontConfig(size=20, color="#FF0000", opacity=0.5),
                position_config=PositionConfig(type=PositionType.TOP_LEFT)
            ),
            WatermarkLayer(
                type=WatermarkType.QRCODE,
                content="https://example.com",
                size=80,
                opacity=0.6,
                position_config=PositionConfig(type=PositionType.BOTTOM_RIGHT)
            ),
            WatermarkLayer(
                type=WatermarkType.TEXT,
                content="Layer 3",
                font_config=FontConfig(size=24, color="#0000FF", opacity=0.4),
                position_config=PositionConfig(type=PositionType.CENTER)
            )
        ]
        
        result = self.processor.apply_multiple_watermarks(self.test_image, layers)
        
        assert result is not None
        assert result.size == self.test_image.size
        assert result.mode == 'RGBA'
    
    def test_apply_multiple_watermarks_with_disabled_layer(self):
        """测试包含禁用层的多层水印"""
        layers = [
            WatermarkLayer(
                type=WatermarkType.TEXT,
                content="Enabled Layer",
                font_config=FontConfig(size=20),
                enabled=True
            ),
            WatermarkLayer(
                type=WatermarkType.TEXT,
                content="Disabled Layer",
                font_config=FontConfig(size=20),
                enabled=False
            )
        ]
        
        result = self.processor.apply_multiple_watermarks(self.test_image, layers)
        
        assert result is not None
        assert result.size == self.test_image.size
    
    def test_color_parsing(self):
        """测试颜色解析"""
        # 测试十六进制颜色
        color = self.processor._parse_color("#FF0000", 0.5)
        assert color == (255, 0, 0, 127)
        
        color = self.processor._parse_color("#00FF00", 1.0)
        assert color == (0, 255, 0, 255)
        
        # 测试无效颜色（应该返回黑色）
        color = self.processor._parse_color("invalid", 0.5)
        assert color == (0, 0, 0, 127)
    
    def test_position_calculation(self):
        """测试位置计算"""
        image_size = (800, 600)
        watermark_size = (100, 50)
        
        # 测试居中位置
        position_config = PositionConfig(type=PositionType.CENTER)
        x, y = self.processor._calculate_position(
            image_size, 
            watermark_size, 
            position_config
        )
        assert x == 350  # (800 - 100) / 2
        assert y == 275  # (600 - 50) / 2
        
        # 测试左上角位置
        position_config = PositionConfig(type=PositionType.TOP_LEFT)
        x, y = self.processor._calculate_position(
            image_size, 
            watermark_size, 
            position_config
        )
        assert x == 10
        assert y == 10
        
        # 测试自定义位置
        position_config = PositionConfig(type=PositionType.CUSTOM, x=100, y=200)
        x, y = self.processor._calculate_position(
            image_size, 
            watermark_size, 
            position_config
        )
        assert x == 100
        assert y == 200
    
    def test_opacity_adjustment(self):
        """测试透明度调整"""
        # 创建测试图像
        test_img = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        
        # 调整透明度
        adjusted = self.processor._adjust_opacity(test_img, 0.5)
        
        assert adjusted.mode == 'RGBA'
        # 检查alpha通道是否被正确调整
        alpha_channel = adjusted.split()[-1]
        alpha_array = np.array(alpha_channel)
        # 透明度应该大约是原来的一半
        assert np.all(alpha_array < 255)


class TestWatermarkConfigParser:
    """水印配置解析器测试类"""
    
    def test_parse_simple_text_watermark(self):
        """测试解析简单文字水印配置"""
        config = {
            "visible_watermark": {
                "enabled": True,
                "layers": [
                    {
                        "type": "text",
                        "content": "Test Watermark",
                        "font": {
                            "family": "Arial",
                            "size": 24,
                            "color": "#FF0000",
                            "opacity": 0.5
                        },
                        "position": {
                            "x": "center",
                            "y": "center"
                        }
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        assert len(layers) == 1
        layer = layers[0]
        assert layer.type == WatermarkType.TEXT
        assert layer.content == "Test Watermark"
        assert layer.font_config.family == "Arial"
        assert layer.font_config.size == 24
        assert layer.font_config.color == "#FF0000"
        assert layer.position_config.type == PositionType.CENTER
    
    def test_parse_qrcode_watermark(self):
        """测试解析二维码水印配置"""
        config = {
            "visible_watermark": {
                "enabled": True,
                "layers": [
                    {
                        "type": "qrcode",
                        "content": "https://example.com",
                        "size": 100,
                        "opacity": 0.8,
                        "position": {
                            "x": "right",
                            "y": "bottom"
                        }
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        assert len(layers) == 1
        layer = layers[0]
        assert layer.type == WatermarkType.QRCODE
        assert layer.content == "https://example.com"
        assert layer.size == 100
        assert layer.opacity == 0.8
        assert layer.position_config.type == PositionType.BOTTOM_RIGHT
    
    def test_parse_multiple_layers(self):
        """测试解析多层水印配置"""
        config = {
            "visible_watermark": {
                "enabled": True,
                "layers": [
                    {
                        "type": "text",
                        "content": "Text Layer",
                        "enabled": True
                    },
                    {
                        "type": "qrcode",
                        "content": "QR Data",
                        "enabled": False
                    },
                    {
                        "type": "barcode",
                        "content": "Barcode Data"
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        assert len(layers) == 3
        assert layers[0].type == WatermarkType.TEXT
        assert layers[0].enabled == True
        assert layers[1].type == WatermarkType.QRCODE
        assert layers[1].enabled == False
        assert layers[2].type == WatermarkType.BARCODE
        assert layers[2].enabled == True  # 默认值
    
    def test_parse_disabled_watermark(self):
        """测试解析禁用的水印配置"""
        config = {
            "visible_watermark": {
                "enabled": False,
                "layers": [
                    {
                        "type": "text",
                        "content": "Should not be parsed"
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        assert len(layers) == 0
    
    def test_parse_custom_position(self):
        """测试解析自定义位置配置"""
        config = {
            "visible_watermark": {
                "enabled": True,
                "layers": [
                    {
                        "type": "text",
                        "content": "Custom Position",
                        "position": {
                            "x": 100,
                            "y": 200,
                            "rotation": 45.0,
                            "spacing": {
                                "horizontal": 150,
                                "vertical": 100
                            }
                        }
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        assert len(layers) == 1
        layer = layers[0]
        assert layer.position_config.type == PositionType.CUSTOM
        assert layer.position_config.x == 100
        assert layer.position_config.y == 200
        assert layer.position_config.rotation == 45.0
        assert layer.position_config.spacing_x == 150
        assert layer.position_config.spacing_y == 100
    
    def test_parse_invalid_config(self):
        """测试解析无效配置"""
        config = {
            "visible_watermark": {
                "enabled": True,
                "layers": [
                    {
                        "type": "invalid_type",
                        "content": "Invalid"
                    }
                ]
            }
        }
        
        layers = WatermarkConfigParser.parse_watermark_config(config)
        
        # 无效配置应该被忽略
        assert len(layers) == 0


if __name__ == "__main__":
    pytest.main([__file__])