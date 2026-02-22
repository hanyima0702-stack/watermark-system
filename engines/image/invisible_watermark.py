import cv2
import numpy as np
import time
import logging
from .config import WatermarkConfig, EmbedResult, ExtractionResult
from .encoding.ecc_encoder import ECCEncoder
from .encoding.scrambler import Scrambler
from .embedding.ppm_modulator import DCTModulator
from .embedding.image_embedder import ImageEmbedder
from .utils.m_sequence import MSequenceGenerator
logger = logging.getLogger(__name__)


class InvisibleWatermarkProcessor:
    def __init__(self, config: WatermarkConfig = None):
        self.config = config if config else WatermarkConfig()
        self.ecc_encoder = ECCEncoder(n=self.config.ecc_n, k=self.config.ecc_k)
        self.scrambler = Scrambler(seed=self.config.scramble_seed)
        self.modulator = DCTModulator(strength=self.config.modulation_strength)

        # 核心修改 3：必须与嵌入端保持一致，改为 5 阶
        self.m_gen = MSequenceGenerator(degree=5)
        self.image_embedder = ImageEmbedder(modulator=self.modulator, watermarkConfig=self.config)

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

    def _get_exact_angle_using_cross_correlation(self, image: np.ndarray):
        if not self.config.enable_spatial_anchors:
            return image, 0, 1.0

        h, w = image.shape[:2]

        # ==========================================
        # 核心改动：只从 U 通道提取雷达信号，避开 Y 通道的 DCT 干扰
        # ==========================================
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        u_channel = image_yuv[:, :, 1].astype(np.float32)
        u_channel -= np.mean(u_channel)

        # 裁剪中心区域加速搜索
        center_y, center_x = h // 2, w // 2
        crop_size = min(512, min(h, w))
        y1, y2 = center_y - crop_size // 2, center_y + crop_size // 2
        x1, x2 = center_x - crop_size // 2, center_x + crop_size // 2
        search_crop = u_channel[y1:y2, x1:x2]

        # 重新生成一致的模板
        rng = np.random.RandomState(42)
        core_pattern = np.sign(rng.randn(31, 31)).astype(np.float32)
        tpl_size = self.config.anchor_spacing
        template = cv2.resize(core_pattern, (tpl_size, tpl_size), interpolation=cv2.INTER_NEAREST)

        # 验证 0 度 Fast Path
        res_0 = cv2.matchTemplate(search_crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val_0, _, _ = cv2.minMaxLoc(res_0)

        if max_val_0 > 0.6:
            logger.info(f"Fast path hit: 0° detected with high correlation {max_val_0:.4f}")
            return image, 0, max_val_0

        max_correlation = max_val_0
        best_angle = 0
        crop_h, crop_w = search_crop.shape
        crop_center = (crop_w // 2, crop_h // 2)

        # 全角度扫描
        for angle in range(1, 360, 1):
            M_rot = cv2.getRotationMatrix2D(crop_center, angle, 1.0)
            rotated_crop = cv2.warpAffine(search_crop, M_rot, (crop_w, crop_h))

            res = cv2.matchTemplate(rotated_crop, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > max_correlation:
                max_correlation = max_val
                best_angle = angle

        logger.info(f"Detected arbitrary angle: {best_angle}° with correlation {max_correlation:.4f}")

        if max_correlation < 0.15:
            logger.warning("Weak correlation, fallback to original image.")
            return image, 0, max_correlation

        # 反向无损转正整张图像
        if best_angle != 0:
            orig_center = (w // 2, h // 2)
            M_correct = cv2.getRotationMatrix2D(orig_center, best_angle, 1.0)

            cos = np.abs(M_correct[0, 0])
            sin = np.abs(M_correct[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M_correct[0, 2] += (new_w / 2) - orig_center[0]
            M_correct[1, 2] += (new_h / 2) - orig_center[1]

            corrected_image = cv2.warpAffine(
                image, M_correct, (new_w, new_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        else:
            corrected_image = image

        return corrected_image, best_angle, max_correlation

    def extract_watermark(self, image_path: str) -> ExtractionResult:
        """
        极速抗攻击提取：互相关定向 + 分辨率扫描
        已去除低效的暴力角度扫描。
        """
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None: return ExtractionResult(success=False, error_message="Image not found")

        # === 1. 新增步骤：利用 m-序列互相关快速纠正 90/180/270 度旋转 ===
        # 这一步极快，并且能有效抵抗摩尔纹干扰
        img_oriented, detected_angle, corr_score = self._get_exact_angle_using_cross_correlation(image)

        rot_h, rot_w = img_oriented.shape[:2]

        # === 2. 准备分辨率搜索列表 (保持原逻辑，针对缩放攻击) ===
        search_widths = set()
        base_w = rot_w
        # 优先搜索原始分辨率
        search_widths.add(base_w)
        # 添加一些常见的缩放比例，减少搜索空间
        for scale in [0.5, 0.75, 1.5, 2.0]:
            search_widths.add(int(base_w * scale))

        # 如果还需要更密集的搜索，保留原来的逻辑
        current_w = self.config.min_width
        while current_w <= self.config.max_width:
            search_widths.add(current_w)
            current_w += self.config.search_step

        # 按距离 base_w 的远近排序
        sorted_widths = sorted(list(search_widths), key=lambda x: abs(x - base_w))
        # 去重并过滤过小的尺寸
        sorted_widths = sorted(list(set([w for w in sorted_widths if w >= 128])))

        best_confidence = 0.0
        logger.info(f"Starting extraction based on oriented image. Scanning {len(sorted_widths)} resolutions.")

        # === 只保留分辨率循环，去除了角度循环 ===
        for target_w in sorted_widths:
            scale = target_w / rot_w
            target_h = int(rot_h * scale)

            # 缩放图像
            # 旧代码：
            # interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            # img_search = cv2.resize(img_oriented, (target_w, target_h), interpolation=interpolation)

            # 新代码：
            if abs(scale - 1.0) < 1e-5:
                img_search = img_oriented.copy()  # 1.0 比例绝对禁止过 resize 算子
            else:
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                img_search = cv2.resize(img_oriented, (target_w, target_h), interpolation=interpolation)

            y_full = cv2.cvtColor(img_search, cv2.COLOR_BGR2YUV)[:, :, 0]

            # === 网格偏移扫描 (保持不变，处理剪裁和相位错位) ===
            step = 2
            for dy in range(0, 8, step):
                for dx in range(0, 8, step):
                    if dy > 0 or dx > 0:
                        y_crop = y_full[dy:, dx:]
                    else:
                        y_crop = y_full

                    # 提取 DCT 差值
                    raw_diffs, rows, cols = self.modulator.demodulate(y_crop)
                    if len(raw_diffs) < 128: continue

                    # 极速剪枝
                    avg_signal = np.mean(np.abs(raw_diffs))
                    if avg_signal < 4.0: continue

                    # 软投票解码 (确保你使用了包含二维循环移位的版本)
                    decoded, success, score = self._decode_soft_with_retry(raw_diffs, rows, cols)

                    if success:
                        logger.info(
                            f"FOUND! Base Angle: {detected_angle}°, Res: {target_w}x{target_h} (Scale: {scale:.2f}), Offset: ({dx},{dy}), Score: {score:.2f}")
                        return ExtractionResult(
                            success=True,
                            watermark_data=decoded,
                            confidence=score,
                            detected_scale=scale,
                            # 这里返回的是相对于转正后图像的偏移
                            grid_offset=(dx, dy),
                            processing_time=time.time() - start_time
                        )

                    if score > best_confidence:
                        best_confidence = score

        return ExtractionResult(success=False, confidence=best_confidence,
                                error_message=f"Extraction failed after rapid scan. Anchor corr: {corr_score:.2f}",
                                processing_time=time.time() - start_time)

    def _decode_soft_with_retry(self, raw_diffs, rows, cols):
        """
        二维软投票解码 + 二维循环移位
        """
        grid_h, grid_w = 16, 8

        # 1. 使用 2D 矩阵进行投票和计数
        votes = np.zeros((grid_h, grid_w), dtype=np.float32)
        counts = np.zeros((grid_h, grid_w), dtype=np.int32)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                # === 核心改进：将线性提取出的差值折叠回二维网格 ===
                block_r = r % grid_h
                block_c = c % grid_w

                votes[block_r, block_c] += raw_diffs[idx]
                counts[block_r, block_c] += 1
                idx += 1

        # 2. 归一化处理
        counts[counts == 0] = 1
        votes = votes / counts

        # 3. 生成候选的二维 Bit 矩阵
        candidate_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        candidate_grid[votes > 0] = 1

        confidence = np.mean(np.abs(votes))

        if confidence < 8.0:
            return None, False, confidence

        # 4. 二维循环移位搜索 (解决任意起点的剪裁导致的数据平移)
        # 遍历所有可能的 (行偏移, 列偏移) 组合
        for shift_r in range(grid_h):
            for shift_c in range(grid_w):
                # 同时在行(axis=0)和列(axis=1)上进行循环移位
                shifted_grid = np.roll(candidate_grid, shift=(-shift_r, -shift_c), axis=(0, 1))

                # 展平回 1D 的 128 位数组送入解码器
                shifted_flat = shifted_grid.flatten()

                try:
                    descrambled = self.scrambler.descramble(shifted_flat)
                    data, success = self.ecc_encoder.decode(descrambled)
                    if success:
                        return data, True, confidence
                except:
                    continue

        return None, False, confidence