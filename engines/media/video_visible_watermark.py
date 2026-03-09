"""
视频明水印处理器
使用FFmpeg实现视频overlay滤镜，支持动态水印叠加
"""
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tempfile
import logging

logger = logging.getLogger(__name__)


class VideoVisibleWatermarkConfig:
    """视频明水印配置"""
    
    def __init__(
        self,
        watermark_path: str,
        position: str = "top-right",
        opacity: float = 0.5,
        scale: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        x_offset: int = 10,
        y_offset: int = 10
    ):
        """
        Args:
            watermark_path: 水印图像路径
            position: 水印位置 (top-left, top-right, bottom-left, bottom-right, center)
            opacity: 透明度 (0.0-1.0)
            scale: 缩放比例 (None表示不缩放)
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            x_offset: X轴偏移量（像素）
            y_offset: Y轴偏移量（像素）
        """
        self.watermark_path = watermark_path
        self.position = position
        self.opacity = max(0.0, min(1.0, opacity))
        self.scale = scale
        self.start_time = start_time
        self.end_time = end_time
        self.x_offset = x_offset
        self.y_offset = y_offset


class VideoVisibleWatermarkProcessor:
    """视频明水印处理器"""
    
    def __init__(self):
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查FFmpeg是否可用"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def add_watermark(
        self,
        input_video: str,
        output_video: str,
        config: VideoVisibleWatermarkConfig,
        video_codec: str = 'libx264',
        audio_codec: str = 'aac',
        crf: int = 18,
        preset: str = 'medium'
    ) -> bool:
        """
        为视频添加明水印
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            config: 水印配置
            video_codec: 视频编码器
            crf: 视频质量 (0-51, 越小质量越好)
            preset: 编码预设 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            audio_codec: 音频编码器
            
        Returns:
            是否成功
        """
        try:
            # 构建FFmpeg命令
            cmd = self._build_ffmpeg_command(
                input_video,
                output_video,
                config,
                video_codec,
                audio_codec,
                crf,
                preset
            )
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # 执行FFmpeg命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Watermark added successfully to {output_video}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to add watermark: {e.stderr}")
        except Exception as e:
            logger.error(f"Error adding watermark: {str(e)}")
            raise
    
    def _build_ffmpeg_command(
        self,
        input_video: str,
        output_video: str,
        config: VideoVisibleWatermarkConfig,
        video_codec: str,
        audio_codec: str,
        crf: int,
        preset: str
    ) -> list:
        """构建FFmpeg命令"""
        cmd = ['ffmpeg', '-i', input_video, '-i', config.watermark_path]
        
        # 构建overlay滤镜
        filter_complex = self._build_overlay_filter(config)
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-c:v', video_codec,
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', audio_codec,
            '-y',  # 覆盖输出文件
            output_video
        ])
        
        return cmd
    
    def _build_overlay_filter(self, config: VideoVisibleWatermarkConfig) -> str:
        """构建overlay滤镜字符串"""
        filters = []
        
        # 缩放水印
        if config.scale is not None:
            filters.append(f"[1:v]scale=iw*{config.scale}:ih*{config.scale}[wm]")
            watermark_input = "[wm]"
        else:
            watermark_input = "[1:v]"
        
        # 设置透明度
        if config.opacity < 1.0:
            alpha_filter = f"{watermark_input}format=rgba,colorchannelmixer=aa={config.opacity}[wm_alpha]"
            filters.append(alpha_filter)
            watermark_input = "[wm_alpha]"
        
        # 计算位置
        position_expr = self._get_position_expression(config)
        
        # 构建overlay
        overlay_filter = f"[0:v]{watermark_input}overlay={position_expr}"
        
        # 添加时间范围控制
        if config.start_time is not None or config.end_time is not None:
            enable_expr = self._get_enable_expression(config)
            overlay_filter += f":enable='{enable_expr}'"
        
        filters.append(overlay_filter)
        
        return ";".join(filters)
    
    def _get_position_expression(self, config: VideoVisibleWatermarkConfig) -> str:
        """获取位置表达式"""
        position_map = {
            'top-left': f'{config.x_offset}:{config.y_offset}',
            'top-right': f'W-w-{config.x_offset}:{config.y_offset}',
            'bottom-left': f'{config.x_offset}:H-h-{config.y_offset}',
            'bottom-right': f'W-w-{config.x_offset}:H-h-{config.y_offset}',
            'center': '(W-w)/2:(H-h)/2'
        }
        
        return position_map.get(config.position, position_map['top-right'])
    
    def _get_enable_expression(self, config: VideoVisibleWatermarkConfig) -> str:
        """获取时间范围表达式"""
        conditions = []
        
        if config.start_time is not None:
            conditions.append(f"gte(t,{config.start_time})")
        
        if config.end_time is not None:
            conditions.append(f"lte(t,{config.end_time})")
        
        if conditions:
            return "*".join(conditions)
        
        return "1"
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def add_text_watermark(
        self,
        input_video: str,
        output_video: str,
        text: str,
        position: str = "top-right",
        font_size: int = 24,
        font_color: str = "white",
        opacity: float = 0.5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        video_codec: str = 'libx264',
        audio_codec: str = 'aac',
        crf: int = 18,
        preset: str = 'medium'
    ) -> bool:
        """
        为视频添加文字水印
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            text: 水印文字
            position: 位置
            font_size: 字体大小
            font_color: 字体颜色
            opacity: 透明度
            start_time: 开始时间
            end_time: 结束时间
            video_codec: 视频编码器
            audio_codec: 音频编码器
            crf: 视频质量
            preset: 编码预设
            
        Returns:
            是否成功
        """
        try:
            # 构建drawtext滤镜
            filter_str = self._build_drawtext_filter(
                text, position, font_size, font_color, opacity,
                start_time, end_time
            )
            
            cmd = [
                'ffmpeg',
                '-i', input_video,
                '-vf', filter_str,
                '-c:v', video_codec,
                '-crf', str(crf),
                '-preset', preset,
                '-c:a', audio_codec,
                '-y',
                output_video
            ]
            
            logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Text watermark added successfully to {output_video}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to add text watermark: {e.stderr}")
        except Exception as e:
            logger.error(f"Error adding text watermark: {str(e)}")
            raise
    
    def _build_drawtext_filter(
        self,
        text: str,
        position: str,
        font_size: int,
        font_color: str,
        opacity: float,
        start_time: Optional[float],
        end_time: Optional[float]
    ) -> str:
        """构建drawtext滤镜"""
        # 转义特殊字符
        text = text.replace(":", r"\:")
        text = text.replace("'", r"\'")
        
        # 位置映射
        position_map = {
            'top-left': 'x=10:y=10',
            'top-right': 'x=w-tw-10:y=10',
            'bottom-left': 'x=10:y=h-th-10',
            'bottom-right': 'x=w-tw-10:y=h-th-10',
            'center': 'x=(w-tw)/2:y=(h-th)/2'
        }
        
        pos_expr = position_map.get(position, position_map['top-right'])
        
        # 构建滤镜
        filter_parts = [
            f"drawtext=text='{text}'",
            f"fontsize={font_size}",
            f"fontcolor={font_color}@{opacity}",
            pos_expr
        ]
        
        # 添加时间范围
        if start_time is not None or end_time is not None:
            enable_conditions = []
            if start_time is not None:
                enable_conditions.append(f"gte(t,{start_time})")
            if end_time is not None:
                enable_conditions.append(f"lte(t,{end_time})")
            
            if enable_conditions:
                enable_expr = "*".join(enable_conditions)
                filter_parts.append(f"enable='{enable_expr}'")
        
        return ":".join(filter_parts)
