import cv2
import time
import tempfile
import os
import shutil
import subprocess
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Dict
from pathlib import Path
import logging

from engines.image import WatermarkConfig, InvisibleWatermarkProcessor

logger = logging.getLogger(__name__)


# =====================================================================
# 多进程 Worker 函数 (必须定义在类外部)
# =====================================================================

def _worker_embed_batch(args):
    """嵌入进程的 Worker：处理一小批帧。"""
    frames_info, scrambled_bits, config_dict = args
    config = WatermarkConfig(**config_dict)
    processor = InvisibleWatermarkProcessor(config)
    processed_frames = []
    for frame, needs_embed in frames_info:
        if needs_embed:
            w_frame = processor.image_embedder.embed(frame, scrambled_bits)
            processed_frames.append(w_frame)
        else:
            processed_frames.append(frame)
    return processed_frames


def _worker_extract_batch(args):
    """提取进程的 Worker：处理一小批帧。"""
    frames_info, config_dict = args
    config = WatermarkConfig(**config_dict)
    processor = InvisibleWatermarkProcessor(config)
    results = []
    for frame, f_idx, t_sec in frames_info:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_img_path = tmp_file.name
        try:
            cv2.imwrite(temp_img_path, frame)
            result = processor.extract_watermark(temp_img_path)
            results.append({
                "frame_index": f_idx,
                "time_sec": t_sec,
                "success": result.success,
                "data": result.watermark_data if result.success else None,
                "confidence": result.confidence
            })
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    return results


# =====================================================================
# 主体类定义
# =====================================================================

class VideoWatermarker:
    """视频暗水印处理工具 (支持按时间范围局部嵌入与提取 + 多进程并发提速)"""

    def __init__(self, config: WatermarkConfig = None):
        if config is None:
            config = WatermarkConfig(
                modulation_strength=30,
                enable_spatial_anchors=True,
                anchor_strength=1.5
            )
        self.config = config
        self.processor = InvisibleWatermarkProcessor(self.config)

    def embed_video_timerange(self, input_video_path: str, output_video_path: str,
                              watermark_text: str, start_time: float,
                              end_time: Optional[float] = None):
        """在视频的指定时间范围内（秒）逐帧嵌入水印。"""
        logger.info(f"开始处理视频: {input_video_path}")
        start_process_time = time.time()

        encoded_bits = self.processor.ecc_encoder.encode(watermark_text)
        scrambled_bits = self.processor.scrambler.scramble(encoded_bits)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开输入视频文件: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        if end_time is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(int(end_time * fps), total_frames - 1)

        if start_frame >= total_frames:
            raise ValueError(f"指定的开始时间 ({start_time}s) 超出了视频总时长。")

        # 使用无损中间格式保留水印信号，避免有损编码破坏频域水印
        # 先写 AVI (FFV1 无损)，处理完后用 FFmpeg 转高质量 H.264
        lossless_tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(lossless_tmp, fourcc, fps, (width, height))

        embedded_count = 0
        current_frame_idx = 0
        # 限制并发进程数，避免内存爆炸（每个进程都要加载水印处理器）
        cpu_cores = min(multiprocessing.cpu_count(), 4)
        chunk_size = 30
        config_dict = self.config.__dict__

        logger.info(f"目标嵌入区间: {start_time}s - {end_time}s (帧 {start_frame} 到 {end_frame}), 总帧数 {total_frames}")

        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            while cap.isOpened():
                buffer_info = []
                for _ in range(chunk_size * cpu_cores):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    needs_embed = (start_frame <= current_frame_idx <= end_frame)
                    if needs_embed:
                        embedded_count += 1
                    buffer_info.append((frame, needs_embed))
                    current_frame_idx += 1

                if not buffer_info:
                    break

                # 每处理一批打印进度
                pct = current_frame_idx / max(total_frames, 1) * 100
                logger.info(f"水印嵌入进度: {current_frame_idx}/{total_frames} 帧 ({pct:.1f}%)")

                chunks = [buffer_info[i:i + chunk_size]
                          for i in range(0, len(buffer_info), chunk_size)]
                tasks = [(chunk, scrambled_bits, config_dict) for chunk in chunks]
                results = executor.map(_worker_embed_batch, tasks)

                for chunk_result in results:
                    for w_frame in chunk_result:
                        out.write(w_frame)

        cap.release()
        out.release()
        logger.info(f"帧嵌入完成，共 {embedded_count} 帧。开始 FFmpeg 编码...")

        # 用 FFmpeg 将无损 AVI 转为高质量 H.264 MP4 (CRF 18 近无损)
        # CRF 值越低质量越高：0=无损, 16=视觉无损, 23=默认, 51=最差
        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd:
            try:
                # 如果原视频有音轨，从原视频复制音频流
                cmd = [
                    ffmpeg_cmd, "-y",
                    "-i", lossless_tmp,
                    "-i", input_video_path,
                    "-map", "0:v:0",       # 视频流来自水印后的无损文件
                    "-map", "1:a?",        # 音频流来自原视频（如果有的话）
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "medium",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    output_video_path,
                ]
                subprocess.run(cmd, capture_output=True, timeout=3600, check=True)
                logger.info("FFmpeg 高质量编码完成: CRF=18, 已保留原音轨")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.warning(f"FFmpeg 转码失败，回退到无损 AVI: {e}")
                shutil.copy2(lossless_tmp, output_video_path)
        else:
            # 没有 FFmpeg，直接用无损 AVI（文件会很大但水印完整）
            logger.warning("未找到 FFmpeg，输出无损 AVI 格式")
            shutil.copy2(lossless_tmp, output_video_path)

        # 清理无损中间文件
        try:
            os.unlink(lossless_tmp)
        except OSError:
            pass

        logger.info(f"视频处理完成！共嵌入 {embedded_count} 帧。耗时: {time.time() - start_process_time:.2f} 秒.")

    def extract_from_timerange(self, video_path: str, start_time: float,
                               end_time: Optional[float] = None) -> List[Dict]:
        """从视频的指定时间范围内（秒）的每一帧中提取水印。"""
        logger.info(f"正在从 {video_path} 提取时间段 {start_time}s - {end_time}s 的水印...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        if end_time is None:
            end_frame = total_frames - 1
        else:
            end_frame = min(int(end_time * fps), total_frames - 1)

        if start_frame >= total_frames:
            return []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame
        frames_to_process = end_frame - start_frame + 1

        results = []
        success_count = 0

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_img_path = tmp_file.name

        try:
            while cap.isOpened() and current_frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imwrite(temp_img_path, frame)
                result = self.processor.extract_watermark(temp_img_path)

                frame_info = {
                    "frame_index": current_frame_idx,
                    "time_sec": current_frame_idx / fps,
                    "success": result.success,
                    "data": result.watermark_data if result.success else None,
                    "confidence": result.confidence
                }
                results.append(frame_info)

                if result.success:
                    success_count += 1

                current_frame_idx += 1
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        cap.release()

        logger.info(f"提取完成: 检查 {frames_to_process} 帧, 成功 {success_count} 帧")

        if success_count > 0:
            extracted_texts = [r["data"] for r in results if r["success"]]
            most_common = max(set(extracted_texts), key=extracted_texts.count)
            logger.info(f"最终高置信度水印内容推断: {most_common}")

        return results


# =====================================================================
# 上层处理器（供 API 路由调用）
# =====================================================================

class VideoWatermarkProcessor:
    """视频水印上层处理器：整合明/暗水印，最终上传 MinIO。"""

    def __init__(self, config: WatermarkConfig = None):
        self.watermarker = VideoWatermarker(config)

    async def process_watermark(
        self, input_path, minio_service,
        invisible_watermark=None, visible_config=None,
        object_key=None, bucket_name=None,
    ):
        import time
        import uuid
        start_time = time.time()
        tmp_files = []
        try:
            current_path = input_path

            # 1. 嵌入暗水印
            if invisible_watermark:
                suffix = Path(input_path).suffix or '.mp4'
                tmp_out = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
                tmp_files.append(tmp_out)
                self.watermarker.embed_video_timerange(
                    input_video_path=current_path,
                    output_video_path=tmp_out,
                    watermark_text=invisible_watermark,
                    start_time=0,
                )
                current_path = tmp_out

            # 2. 嵌入明水印（FFmpeg）
            if visible_config is not None:
                from .video_visible_watermark import VideoVisibleWatermarkProcessor
                ext = Path(input_path).suffix or '.mp4'
                tmp_visible = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
                tmp_files.append(tmp_visible)
                VideoVisibleWatermarkProcessor().add_watermark(
                    current_path, tmp_visible, visible_config
                )
                current_path = tmp_visible

            # 3. 上传 MinIO
            with open(current_path, 'rb') as f:
                video_bytes = f.read()

            ext = Path(input_path).suffix.lstrip('.') or 'mp4'
            target_bucket = bucket_name or minio_service.config.result_bucket
            target_key = object_key or f"watermarked/{uuid.uuid4().hex}.{ext}"

            await minio_service.upload_file(
                bucket_name=target_bucket, object_key=target_key,
                file_data=video_bytes, content_type=f"video/{ext}",
                metadata={"watermark": invisible_watermark or ""},
            )

            return {
                "success": True,
                "minio_object_key": f"{target_bucket}/{target_key}",
                "processing_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"视频水印处理失败: {e}")
            return {"success": False, "error": str(e)}
        finally:
            for tmp in tmp_files:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
