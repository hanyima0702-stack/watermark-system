import cv2
import numpy as np
import time
import logging
from .config import WatermarkConfig, EmbedResult, ExtractionResult
from .encoding.ecc_encoder import ECCEncoder
from .encoding.scrambler import Scrambler
from .embedding.ppm_modulator import DCTModulator
from .embedding.image_embedder import ImageEmbedder

logger = logging.getLogger(__name__)


class InvisibleWatermarkProcessor:
    def __init__(self, config: WatermarkConfig = None):
        self.config = config if config else WatermarkConfig()
        self.ecc_encoder = ECCEncoder(n=self.config.ecc_n, k=self.config.ecc_k)
        self.scrambler = Scrambler(seed=self.config.scramble_seed)
        self.modulator = DCTModulator(strength=self.config.modulation_strength)
        self.image_embedder = ImageEmbedder(modulator=self.modulator)

    def embed_watermark(self, image_path: str, watermark: str, output_path: str) -> EmbedResult:
        # 嵌入代码保持不变，请直接使用上一个版本的 embed_watermark
        # 重点在于使用新的 DCTModulator (低频)
        start_time = time.time()
        try:
            image = cv2.imread(image_path)
            if image is None: raise ValueError("Image not found")
            encoded_bits = self.ecc_encoder.encode(watermark)
            scrambled_bits = self.scrambler.scramble(encoded_bits)
            watermarked_img = self.image_embedder.embed(image, scrambled_bits)
            cv2.imwrite(output_path, watermarked_img)
            return EmbedResult(
                success=True, watermark_data=watermark, encoded_data="",
                block_count=(0, 0), processing_time=time.time() - start_time,
                image_size=image.shape[:2]
            )
        except Exception as e:
            return EmbedResult(success=False, error_message=str(e), watermark_data="", encoded_data="",
                               block_count=(0, 0), processing_time=0, image_size=(0, 0))

    def extract_watermark(self, image_path: str) -> ExtractionResult:
        """
        终极抗攻击提取：分辨率金字塔 + 网格相位搜索 + 软投票
        """
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None: return ExtractionResult(success=False, error_message="Image not found")

        original_h, original_w = image.shape[:2]

        # 1. 生成搜索分辨率列表
        # 包含：原始宽度（抗纯剪裁）、一系列缩放宽度（抗缩放+剪裁）
        # 步长设为 32 或 40，覆盖常见缩放
        search_widths = set()
        search_widths.add(original_w)  # 必须包含原图尺寸

        # 添加缩小和放大的尺寸
        current_w = self.config.min_width
        while current_w <= self.config.max_width:
            search_widths.add(current_w)
            current_w += self.config.search_step

        # 排序：优先搜原图附近，然后向两边扩散
        sorted_widths = sorted(list(search_widths), key=lambda x: abs(x - original_w))

        best_confidence = 0.0

        logger.info(f"Scanning {len(sorted_widths)} resolutions with grid search...")

        # === 外层循环：分辨率 (抗缩放) ===
        for target_w in sorted_widths:
            # 计算等比例高度
            scale = target_w / original_w
            target_h = int(original_h * scale)

            # 如果尺寸太小，跳过
            if target_w < 64 or target_h < 64: continue

            # 缩放图像
            # 缩小用 INTER_AREA (抗噪)，放大用 INTER_LINEAR
            if scale < 1.0:
                img_search = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                img_search = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            y_full = cv2.cvtColor(img_search, cv2.COLOR_BGR2YUV)[:, :, 0]

            # === 内层循环：网格偏移 (抗剪裁/相位错位) ===
            # DCT 块是 8x8 的。剪裁会导致网格错位 0-7 个像素。
            # 我们遍历 (0,0) 到 (7,7) 的偏移。
            # 为了速度，步长设为 2 (即搜 0, 2, 4, 6)，通常足够撞上
            step = 2
            for dy in range(0, 8, step):
                for dx in range(0, 8, step):

                    # 模拟网格偏移：直接切片
                    # y_crop 相当于把原点移动到了 (dx, dy)
                    if dy > 0 or dx > 0:
                        y_crop = y_full[dy:, dx:]
                    else:
                        y_crop = y_full

                    # 提取 DCT 差值 (Soft Decision)
                    # raw_diffs 是浮点数数组，正负值代表 1/0 倾向
                    raw_diffs, rows, cols = self.modulator.demodulate(y_crop)

                    if len(raw_diffs) < 128: continue

                    # === 极速剪枝 (Signal Strength Check) ===
                    # 这是速度的关键！
                    # 如果网格没对齐，或者分辨率不对，提取出来的差值全是随机噪声，平均绝对值很小。
                    # 如果对齐了，由于 strength=50，平均绝对值应该很大(>10)。
                    avg_signal = np.mean(np.abs(raw_diffs))

                    # 阈值：根据 modulation_strength=50 设定。
                    # 如果 avg_signal < 5，说明完全是噪声，直接跳过 ECC 解码
                    if avg_signal < 5.0:
                        continue

                    # === 软投票解码 ===
                    decoded, success, score = self._decode_soft_with_retry(raw_diffs, rows, cols)

                    if success:
                        logger.info(
                            f"FOUND! Resolution: {target_w}x{target_h}, Offset: ({dx},{dy}), Score: {score:.2f}")
                        return ExtractionResult(
                            success=True,
                            watermark_data=decoded,
                            confidence=score,
                            detected_scale=scale,
                            grid_offset=(dx, dy),
                            processing_time=time.time() - start_time
                        )

                    if score > best_confidence:
                        best_confidence = score

        return ExtractionResult(success=False, confidence=best_confidence, error_message="Extraction failed",
                                processing_time=time.time() - start_time)

    def _decode_soft_with_retry(self, raw_diffs, rows, cols):
        """
        软投票解码 + 循环移位
        """
        n = 128
        votes = np.zeros(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)

        # 1. 对角线折叠 + 软累加
        # 即使被剪裁，(r+c) 的相对关系不变
        idx = 0
        for r in range(rows):
            for c in range(cols):
                bit_pos = (r + c) % n
                votes[bit_pos] += raw_diffs[idx]
                counts[bit_pos] += 1
                idx += 1

        # 2. 归一化
        counts[counts == 0] = 1
        votes = votes / counts

        # 3. 生成候选 Bit
        candidate = np.zeros(n, dtype=np.uint8)
        candidate[votes > 0] = 1

        # 计算这一轮的置信度 (平均信号强度)
        confidence = np.mean(np.abs(votes))

        # 4. 循环移位搜索 (解决剪裁导致的整体位移)
        # 只有当信号强度足够时才尝试解码 (再次剪枝)
        if confidence < 8.0:
            return None, False, confidence

        for shift in range(n):
            shifted = np.roll(candidate, -shift)
            try:
                descrambled = self.scrambler.descramble(shifted)
                data, success = self.ecc_encoder.decode(descrambled)
                if success:
                    return data, True, confidence
            except:
                continue

        return None, False, confidence