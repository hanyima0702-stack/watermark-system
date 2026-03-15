import cv2
import numpy as np
import time
import uuid
import io
import logging
from typing import Optional
from PIL import Image
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
        self.image = None

        # 核心修改 3：必须与嵌入端保持一致，改为 5 阶
        self.m_gen = MSequenceGenerator(degree=5)
        self.image_embedder = ImageEmbedder(modulator=self.modulator, watermarkConfig=self.config)
    def read_image(self,image: np.ndarray):
        try:
            self.image = image
        except Exception as e:
            print("图片读取失败")

    def _embed_to_array(self, watermark: str) -> np.ndarray:
        """
        内部方法：嵌入水印并返回图像数组（供 pdf_processor 等内部使用）。
        调用前需先通过 read_image() 加载图片。
        """
        if self.image is None:
            raise ValueError("Image not loaded. Call read_image() first.")
        encoded_bits = self.ecc_encoder.encode(watermark)
        scrambled_bits = self.scrambler.scramble(encoded_bits)
        return self.image_embedder.embed(self.image, scrambled_bits)

    async def process_watermark(
        self,
        image_path: str,
        minio_service,
        invisible_watermark: Optional[str] = None,
        visible_processor=None,
        visible_layers=None,
        object_key: str = None,
        bucket_name: str = None,
    ) -> EmbedResult:
        """
        上层方法：根据参数决定嵌入明/暗水印，最终上传 MinIO。

        Args:
            image_path:          原始图片路径
            minio_service:       MinIOService 实例
            invisible_watermark: 暗水印字符串，None 则跳过
            visible_processor:   VisibleWatermarkProcessor 实例，None 则跳过明水印
            visible_layers:      明水印层列表（WatermarkLayer list）
            object_key:          MinIO 对象键，不传则自动生成
            bucket_name:         目标 bucket，不传则使用 result_bucket
        """
        start_time = time.time()
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")

            # 1. 嵌入暗水印
            if invisible_watermark:
                self.read_image(img)
                img = self.embed(invisible_watermark)

            # 2. 嵌入明水印
            if visible_processor and visible_layers:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil_img = visible_processor.apply_multiple_watermarks(pil_img, visible_layers)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 3. 上传 MinIO
            success, img_buf = cv2.imencode(".png", img)
            if not success:
                raise RuntimeError("Failed to encode image to PNG")

            target_bucket = bucket_name or minio_service.config.result_bucket
            target_key = object_key or f"watermarked/{uuid.uuid4().hex}.png"

            await minio_service.upload_file(
                bucket_name=target_bucket,
                object_key=target_key,
                file_data=img_buf.tobytes(),
                content_type="image/png",
                metadata={"watermark": invisible_watermark or ""},
            )

            return EmbedResult(
                success=True,
                watermark_data=invisible_watermark or "",
                encoded_data="",
                block_count=(0, 0),
                processing_time=time.time() - start_time,
                image_size=img.shape[:2],
                minio_object_key=f"{target_bucket}/{target_key}",
            )
        except Exception as e:
            logger.error(f"process_watermark failed: {e}", exc_info=True)
            raise

    def embed(
        self,
        watermark: str,
        image_path: str = None,
    ) -> np.ndarray:
        """嵌入暗水印，返回含水印的图像数组，不涉及存储。"""
        if image_path is not None:
            self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found. Provide image_path or call read_image() first.")
        encoded_bits = self.ecc_encoder.encode(watermark)
        scrambled_bits = self.scrambler.scramble(encoded_bits)
        return self.image_embedder.embed(self.image, scrambled_bits)

    def _get_exact_angle_using_cross_correlation(self, image: np.ndarray):
        if not self.config.enable_spatial_anchors:
            return image, 0, 1.0

        h, w = image.shape[:2]
        # ==========================================
        # 1. 转灰度
        # ==========================================
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 2. 终极显影：CLAHE (对比度受限自适应直方图均衡)
        # 这一步极其关键！它能把屏摄后被压缩到只剩 1 个像素差的微弱波动，
        # 强行拉伸到 20-30 的色阶差，让雷达重新“看见”锚点。
        # ==========================================
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # ==========================================
        # 3. 高斯模糊去摩尔纹 (作用于增强后的图像)
        # ==========================================
        gray_blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0).astype(np.float32)
        gray_blurred -= np.mean(gray_blurred)

        # 截取超大安全区，必须扩大到 1024 才能装得下被放大的物理模板
        center_y, center_x = h // 2, w // 2
        safe_crop_size = 1024

        y1, y2 = max(0, center_y - safe_crop_size // 2), min(h, center_y + safe_crop_size // 2)
        x1, x2 = max(0, center_x - safe_crop_size // 2), min(w, center_x + safe_crop_size // 2)

        search_crop_large = gray_blurred[y1:y2, x1:x2]

        pad_bottom = safe_crop_size - search_crop_large.shape[0]
        pad_right = safe_crop_size - search_crop_large.shape[1]
        if pad_bottom > 0 or pad_right > 0:
            search_crop_large = cv2.copyMakeBorder(
                search_crop_large, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
            )

        large_h, large_w = search_crop_large.shape
        large_center = (large_w // 2, large_h // 2)

        # 核心图案
        rng = np.random.RandomState(42)
        core_pattern = np.sign(rng.randn(31, 31)).astype(np.float32)

        best_correlation = -1.0
        best_angle = 0
        best_scale = 1.0

        # ==========================================
        # 2. 核心修复：多尺度金字塔扫描 (对抗手机像素缩放)
        # ==========================================
        # 如果手机像素高或者拍得近，照片里的锚点会被放大 (1.5x - 3.0x)
        # 如果距离远，锚点会缩小 (0.5x - 0.75x)
        #scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        scales = [1.0]

        for scale in scales:
            current_tpl_size = int(self.config.anchor_spacing * scale)
            # 模板不能大过我们的搜索框
            if current_tpl_size > 512: continue

            # 根据当前的物理缩放猜测，生成对应大小的钥匙
            template = cv2.resize(core_pattern, (current_tpl_size, current_tpl_size), interpolation=cv2.INTER_NEAREST)

            # 动态计算提取纯净内框的大小 (模板尺寸 + 128像素的滑动位移余量)
            clean_size = current_tpl_size + 128
            start_y = large_center[1] - clean_size // 2
            start_x = large_center[0] - clean_size // 2

            # 极速粗搜：步长改为 5 度，快速定位方向
            for angle in range(0, 360, 5):
                M_rot = cv2.getRotationMatrix2D(large_center, angle, 1.0)
                rotated_large = cv2.warpAffine(
                    search_crop_large, M_rot, (large_w, large_h), borderMode=cv2.BORDER_REPLICATE
                )

                # 完美避开边缘畸变的纯净内框
                clean_crop = rotated_large[start_y:start_y + clean_size, start_x:start_x + clean_size]

                res = cv2.matchTemplate(clean_crop, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_correlation:
                    best_correlation = max_val
                    best_angle = angle
                    best_scale = scale

        logger.info(f"Coarse scan found: Angle {best_angle}°, Scale {best_scale}x with corr {best_correlation:.4f}")

        # ==========================================
        # 3. 精细微调：在最佳尺度下，寻找 1 度的绝对精准角度
        # ==========================================
        if best_correlation > 0.15:
            current_tpl_size = int(self.config.anchor_spacing * best_scale)
            template = cv2.resize(core_pattern, (current_tpl_size, current_tpl_size), interpolation=cv2.INTER_NEAREST)
            clean_size = current_tpl_size + 128
            start_y = large_center[1] - clean_size // 2
            start_x = large_center[0] - clean_size // 2

            fine_best_corr = best_correlation
            fine_best_angle = best_angle

            for angle in range(best_angle - 4, best_angle + 5, 1):
                if angle == best_angle: continue

                M_rot = cv2.getRotationMatrix2D(large_center, angle, 1.0)
                rotated_large = cv2.warpAffine(
                    search_crop_large, M_rot, (large_w, large_h), borderMode=cv2.BORDER_REPLICATE
                )
                clean_crop = rotated_large[start_y:start_y + clean_size, start_x:start_x + clean_size]

                res = cv2.matchTemplate(clean_crop, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > fine_best_corr:
                    fine_best_corr = max_val
                    fine_best_angle = angle

            best_angle = fine_best_angle
            best_correlation = fine_best_corr

        logger.info(f"Final Detected Arbitrary Angle: {best_angle}° with correlation {best_correlation:.4f}")

        if best_correlation < 0.15:
            logger.warning("Weak correlation, fallback to original image.")
            return image, 0, best_correlation

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

        return corrected_image, best_angle, best_correlation

    def extract_watermark(self, image_path: str = None) -> ExtractionResult:
        """
        极速抗攻击提取：互相关定向 + 分辨率扫描
        已去除低效的暴力角度扫描。
        """
        start_time = time.time()
        self.image = cv2.imread(image_path)
        if self.image is None: return ExtractionResult(success=False, error_message="Image not found")

        # === 1. 新增步骤：利用 m-序列互相关快速纠正 90/180/270 度旋转 ===
        # 这一步极快，并且能有效抵抗摩尔纹干扰
        img_oriented, detected_angle, corr_score = self._get_exact_angle_using_cross_correlation(self.image)

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