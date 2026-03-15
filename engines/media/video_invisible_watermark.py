"""
视频暗水印处理器
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import tempfile
import logging

logger = logging.getLogger(__name__)


class VideoInvisibleWatermarkConfig:
    def __init__(self, method='iframe_frequency', strength=0.1, gop_size=12, watermark_bits=None):
        self.method = method
        self.strength = strength
        self.gop_size = gop_size
        self.watermark_bits = watermark_bits


class IFrameFrequencyWatermark:
    def __init__(self, strength=0.1):
        self.strength = strength
        self.seed = 42

    def embed_frame(self, frame, watermark_bits):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0].astype(np.float32)
        h, w = y_channel.shape
        rng = np.random.RandomState(self.seed)
        pattern = rng.randn(h, w).astype(np.float32)
        segment_h = h // len(watermark_bits)
        for i, bit in enumerate(watermark_bits):
            sign = 1.0 if bit == 1 else -1.0
            y_start = i * segment_h
            y_end = min((i + 1) * segment_h, h)
            y_channel[y_start:y_end, :] += sign * self.strength * pattern[y_start:y_end, :]
        yuv[:, :, 0] = np.clip(y_channel, 0, 255).astype(np.uint8)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


class VideoInvisibleWatermarker:
    def __init__(self, config=None):
        self.config = config or VideoInvisibleWatermarkConfig()
        self.embedder = IFrameFrequencyWatermark(strength=self.config.strength)

    def embed_file(self, input_path, watermark_data):
        clean_bits = np.array([int(b) for b in watermark_data if b in ('0', '1')], dtype=np.uint8)
        if len(clean_bits) == 0:
            raise ValueError("Watermark data must contain 0s and 1s.")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        suffix = Path(input_path).suffix or '.mp4'
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(self.config.gop_size, 1) == 0:
                frame = self.embedder.embed_frame(frame, clean_bits)
            out.write(frame)
            frame_idx += 1
        cap.release()
        out.release()
        logger.info(f"Video watermark embedded, {frame_idx} frames")
        return tmp_path


class VideoWatermarkProcessor:
    def __init__(self, invisible_config=None):
        self.invisible_watermarker = VideoInvisibleWatermarker(invisible_config)

    async def process_watermark(self, input_path, minio_service, invisible_watermark=None, visible_config=None, object_key=None, bucket_name=None):
        import time, uuid
        start_time = time.time()
        tmp_files = []
        try:
            current_path = input_path
            if invisible_watermark:
                tmp_invisible = self.invisible_watermarker.embed_file(current_path, invisible_watermark)
                tmp_files.append(tmp_invisible)
                current_path = tmp_invisible
            if visible_config is not None:
                from .video_visible_watermark import VideoVisibleWatermarkProcessor
                ext = Path(input_path).suffix or '.mp4'
                tmp_visible = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_files.append(tmp_visible)
                VideoVisibleWatermarkProcessor().add_watermark(current_path, tmp_visible, visible_config)
                current_path = tmp_visible
            with open(current_path, 'rb') as f:
                video_bytes = f.read()
            ext = Path(input_path).suffix.lstrip('.') or 'mp4'
            target_bucket = bucket_name or minio_service.config.result_bucket
            target_key = object_key or f"watermarked/{uuid.uuid4().hex}.{ext}"
            await minio_service.upload_file(bucket_name=target_bucket, object_key=target_key, file_data=video_bytes, content_type=f"video/{ext}", metadata={"watermark": invisible_watermark or ""})
            return {"success": True, "minio_object_key": f"{target_bucket}/{target_key}", "processing_time": time.time() - start_time}
        except Exception as e:
            logger.error(f"Video watermark failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            for tmp in tmp_files:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass