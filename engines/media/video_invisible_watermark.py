
import cv2
import time
import tempfile
import os
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Optional, List, Dict

# 根据你的实际项目路径导入
from engines.image import WatermarkConfig, InvisibleWatermarkProcessor


# =====================================================================
# 多进程 Worker 函数 (必须定义在类外部)
# =====================================================================
def _worker_embed_batch(args):
    """
    嵌入进程的 Worker：处理一小批帧。
    判断每一帧是否需要嵌入（通过布尔标记），从而避免不必要的计算。
    """
    frames_info, scrambled_bits, config_dict = args
    # 还原配置并实例化处理器
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
    """
    提取进程的 Worker：处理一小批帧。
    每个进程内部独立管理临时文件，避免文件读写冲突。
    """
    frames_info, config_dict = args
    config = WatermarkConfig(**config_dict)
    processor = InvisibleWatermarkProcessor(config)

    results = []
    for frame, f_idx, t_sec in frames_info:
        # 每个进程/每帧创建独立的临时文件
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
    """
    视频暗水印处理工具 (支持按时间范围局部嵌入与提取 + 多进程并发提速)
    """

    def __init__(self, config: WatermarkConfig = None):
        if config is None:
            config = WatermarkConfig(
                modulation_strength=10,
                enable_spatial_anchors=True,
                anchor_strength=2
            )
        self.config = config
        self.processor = InvisibleWatermarkProcessor(self.config)

    def embed_video_timerange(self, input_video_path: str, output_video_path: str,
                              watermark_text: str, start_time: float, end_time: Optional[float] = None):
        """
        在视频的指定时间范围内（秒）逐帧嵌入水印。
        """
        print(f"[*] 开始处理视频: {input_video_path}")
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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        embedded_count = 0
        current_frame_idx = 0

        # 多进程配置
        cpu_cores = multiprocessing.cpu_count()
        chunk_size = 30  # 每个核心每次处理的帧数
        config_dict = self.config.__dict__

        print(
            f"[*] 目标嵌入区间: {start_time}s - {end_time if end_time else '结束'}s (帧 {start_frame} 到 {end_frame})")
        print(f"[*] 启动多进程并发引擎，CPU 核心数: {cpu_cores}")

        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            with tqdm(total=total_frames, desc="视频处理进度") as pbar:
                while cap.isOpened():
                    buffer_info = []
                    # 批量读取塞满内存缓冲
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

                    # 切割成 chunks 分发
                    chunks = [
                        buffer_info[i:i + chunk_size]
                        for i in range(0, len(buffer_info), chunk_size)
                    ]
                    tasks = [(chunk, scrambled_bits, config_dict) for chunk in chunks]

                    # map 会自动保证多进程返回的顺序与提交的顺序一致
                    results = executor.map(_worker_embed_batch, tasks)

                    for chunk_result in results:
                        for w_frame in chunk_result:
                            out.write(w_frame)
                            pbar.update(1)

        cap.release()
        out.release()
        print(f"[*] 视频处理完成！共嵌入 {embedded_count} 帧。耗时: {time.time() - start_process_time:.2f} 秒.")
        print(f"[*] 输出文件保存至: {output_video_path}")

    def extract_from_timerange(self, video_path: str, start_time: float, end_time: Optional[float] = None ) -> List[Dict]:
        """
        从视频的指定时间范围内（秒）的每一帧中提取水印。
        返回一个包含所有帧提取结果的列表。
        """
        print(f"\n[*] 正在从 {video_path} 提取时间段 {start_time}s - {end_time}s 的水印...")
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
            print("[-] 指定的开始时间超出了视频总时长。")
            return []

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame

        frames_to_process = end_frame - start_frame + 1
        print(f"[*] 目标帧范围: {start_frame} 到 {end_frame}，共需处理 {frames_to_process} 帧。")

        results = []
        success_count = 0

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_img_path = tmp_file.name

        try:
            with tqdm(total=frames_to_process, desc="逐帧提取进度") as pbar:
                while cap.isOpened() and current_frame_idx <= end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 无损写入临时文件，供底层提取算法调用
                    cv2.imwrite(temp_img_path, frame)
                    result = self.processor.extract_watermark(temp_img_path)

                    # 记录结果
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
                    pbar.update(1)

        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        # 打印提取摘要
        print("\n[*] 提取任务摘要:")
        print(f"    - 检查总帧数: {frames_to_process}")
        print(f"    - 成功提取数: {success_count}")
        print(f"    - 整体成功率: {(success_count / frames_to_process) * 100:.2f}%")

        # 统计出现次数最多的有效水印内容
        if success_count > 0:
            extracted_texts = [r["data"] for r in results if r["success"]]
            most_common_text = max(set(extracted_texts), key=extracted_texts.count)
            print(f"    => 最终高置信度水印内容推断: 【{most_common_text}】")

        return results

