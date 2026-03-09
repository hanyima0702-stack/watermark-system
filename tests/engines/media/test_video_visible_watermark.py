"""
视频明水印处理器单元测试
"""
import pytest
import os
import tempfile
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

from engines.media.video_visible_watermark import (
    VideoVisibleWatermarkProcessor,
    VideoVisibleWatermarkConfig
)


@pytest.fixture
def processor():
    """创建处理器实例"""
    return VideoVisibleWatermarkProcessor()


@pytest.fixture
def test_video(tmp_path):
    """创建测试视频"""
    video_path = tmp_path / "test_video.mp4"
    
    # 使用FFmpeg创建一个简单的测试视频（5秒，红色背景）
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'color=c=red:s=640x480:d=5',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(video_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return str(video_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("FFmpeg not available")


@pytest.fixture
def test_watermark(tmp_path):
    """创建测试水印图像"""
    watermark_path = tmp_path / "watermark.png"
    
    # 创建一个简单的水印图像（白色文字）
    img = Image.new('RGBA', (200, 50), (255, 255, 255, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 40], fill=(255, 255, 255, 200))
    
    img.save(str(watermark_path))
    return str(watermark_path)


class TestVideoVisibleWatermarkConfig:
    """测试水印配置类"""
    
    def test_config_creation(self, test_watermark):
        """测试配置创建"""
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="top-right",
            opacity=0.5
        )
        
        assert config.watermark_path == test_watermark
        assert config.position == "top-right"
        assert config.opacity == 0.5
    
    def test_opacity_clamping(self, test_watermark):
        """测试透明度限制"""
        config1 = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            opacity=1.5
        )
        assert config1.opacity == 1.0
        
        config2 = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            opacity=-0.5
        )
        assert config2.opacity == 0.0
    
    def test_time_range_config(self, test_watermark):
        """测试时间范围配置"""
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            start_time=1.0,
            end_time=3.0
        )
        
        assert config.start_time == 1.0
        assert config.end_time == 3.0


class TestVideoVisibleWatermarkProcessor:
    """测试视频明水印处理器"""
    
    def test_processor_creation(self, processor):
        """测试处理器创建"""
        assert processor is not None
    
    def test_ffmpeg_check(self):
        """测试FFmpeg检查"""
        try:
            processor = VideoVisibleWatermarkProcessor()
            assert True
        except RuntimeError:
            pytest.skip("FFmpeg not available")
    
    def test_add_watermark_basic(self, processor, test_video, test_watermark, tmp_path):
        """测试基本水印添加"""
        output_video = tmp_path / "output_watermarked.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="top-right",
            opacity=0.5
        )
        
        result = processor.add_watermark(
            test_video,
            str(output_video),
            config
        )
        
        assert result is True
        assert output_video.exists()
        assert output_video.stat().st_size > 0
    
    def test_add_watermark_different_positions(self, processor, test_video, test_watermark, tmp_path):
        """测试不同位置的水印"""
        positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center']
        
        for position in positions:
            output_video = tmp_path / f"output_{position}.mp4"
            
            config = VideoVisibleWatermarkConfig(
                watermark_path=test_watermark,
                position=position,
                opacity=0.7
            )
            
            result = processor.add_watermark(
                test_video,
                str(output_video),
                config
            )
            
            assert result is True
            assert output_video.exists()
    
    def test_add_watermark_with_scale(self, processor, test_video, test_watermark, tmp_path):
        """测试带缩放的水印"""
        output_video = tmp_path / "output_scaled.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="center",
            opacity=0.8,
            scale=0.5
        )
        
        result = processor.add_watermark(
            test_video,
            str(output_video),
            config
        )
        
        assert result is True
        assert output_video.exists()
    
    def test_add_watermark_with_time_range(self, processor, test_video, test_watermark, tmp_path):
        """测试时间范围控制"""
        output_video = tmp_path / "output_timed.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="bottom-right",
            opacity=0.6,
            start_time=1.0,
            end_time=3.0
        )
        
        result = processor.add_watermark(
            test_video,
            str(output_video),
            config
        )
        
        assert result is True
        assert output_video.exists()
    
    def test_add_watermark_quality_settings(self, processor, test_video, test_watermark, tmp_path):
        """测试视频质量设置"""
        output_video = tmp_path / "output_quality.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="top-left"
        )
        
        result = processor.add_watermark(
            test_video,
            str(output_video),
            config,
            crf=23,
            preset='fast'
        )
        
        assert result is True
        assert output_video.exists()
    
    def test_get_video_info(self, processor, test_video):
        """测试获取视频信息"""
        info = processor.get_video_info(test_video)
        
        assert 'format' in info
        assert 'streams' in info
        assert len(info['streams']) > 0
    
    def test_add_text_watermark(self, processor, test_video, tmp_path):
        """测试文字水印"""
        output_video = tmp_path / "output_text.mp4"
        
        result = processor.add_text_watermark(
            test_video,
            str(output_video),
            text="Test Watermark",
            position="bottom-right",
            font_size=24,
            font_color="white",
            opacity=0.7
        )
        
        assert result is True
        assert output_video.exists()
    
    def test_add_text_watermark_with_time_range(self, processor, test_video, tmp_path):
        """测试带时间范围的文字水印"""
        output_video = tmp_path / "output_text_timed.mp4"
        
        result = processor.add_text_watermark(
            test_video,
            str(output_video),
            text="Timed Watermark",
            position="center",
            start_time=1.0,
            end_time=4.0
        )
        
        assert result is True
        assert output_video.exists()
    
    def test_position_expression(self, processor, test_watermark):
        """测试位置表达式生成"""
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            position="top-left",
            x_offset=20,
            y_offset=30
        )
        
        expr = processor._get_position_expression(config)
        assert expr == "20:30"
        
        config.position = "bottom-right"
        expr = processor._get_position_expression(config)
        assert "W-w" in expr and "H-h" in expr
    
    def test_enable_expression(self, processor, test_watermark):
        """测试时间范围表达式"""
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark,
            start_time=2.0,
            end_time=5.0
        )
        
        expr = processor._get_enable_expression(config)
        assert "gte(t,2.0)" in expr
        assert "lte(t,5.0)" in expr
    
    def test_invalid_video_path(self, processor, test_watermark, tmp_path):
        """测试无效视频路径"""
        output_video = tmp_path / "output.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path=test_watermark
        )
        
        with pytest.raises(RuntimeError):
            processor.add_watermark(
                "nonexistent_video.mp4",
                str(output_video),
                config
            )
    
    def test_invalid_watermark_path(self, processor, test_video, tmp_path):
        """测试无效水印路径"""
        output_video = tmp_path / "output.mp4"
        
        config = VideoVisibleWatermarkConfig(
            watermark_path="nonexistent_watermark.png"
        )
        
        with pytest.raises(RuntimeError):
            processor.add_watermark(
                test_video,
                str(output_video),
                config
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
