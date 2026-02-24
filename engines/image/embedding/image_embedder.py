# embedding/image_embedder.py

import cv2
import numpy as np
from .ppm_modulator import DCTModulator
from engines.image.config import WatermarkConfig
# 引入 m-序列生成器 (假设你把它放在了 utils 目录下)
from ..utils.m_sequence import MSequenceGenerator


class ImageEmbedder:
    def __init__(self, modulator: DCTModulator, watermarkConfig: WatermarkConfig = None):
        self.modulator = modulator
        self.watermarkConfig = watermarkConfig if watermarkConfig is not None else WatermarkConfig()
        self._anchor_pattern = None

    def _generate_anchor_pattern(self, spacing: int) -> np.ndarray:
        if self._anchor_pattern is not None: return self._anchor_pattern

        # 1. 放弃复杂的 LFSR，使用确定的随机数种子生成完美的 31x31 正交二维噪声
        rng = np.random.RandomState(42)
        core_pattern = np.sign(rng.randn(31, 31)).astype(np.float32)

        # 2. 放大到 128x128。注意：必须用 INTER_NEAREST（最近邻）！
        # 这样能保证边缘锐利，全是绝对的 1 和 -1，彻底消灭插值产生的灰色糊糊
        patch = cv2.resize(core_pattern, (spacing, spacing), interpolation=cv2.INTER_NEAREST)
        self._anchor_pattern = patch
        return patch

    def embed(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        cfg = self.watermarkConfig

        # ==========================================
        # 1. 铺设强鲁棒性空间锚点 (全通道覆盖)
        # ==========================================
        img_float = image.astype(np.float32)
        if cfg.enable_spatial_anchors and cfg.anchor_strength > 0:
            h, w = image.shape[:2]
            anchor_patch = self._generate_anchor_pattern(cfg.anchor_spacing)
            noise_layer = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, cfg.anchor_spacing):
                for x in range(0, w, cfg.anchor_spacing):
                    patch_h = min(cfg.anchor_spacing, h - y)
                    patch_w = min(cfg.anchor_spacing, w - x)
                    noise_layer[y:y + patch_h, x:x + patch_w] = anchor_patch[:patch_h, :patch_w]

            # 直接加到 BGR 全通道，极其抗色彩压缩
            noise_delta = noise_layer * cfg.anchor_strength
            img_float += noise_delta[:, :, np.newaxis]

        anchored_bgr = np.clip(img_float, 0, 255).astype(np.uint8)

        # ==========================================
        # 2. 转换 YUV 并在 Y 通道执行 DCT
        # ==========================================
        image_yuv = cv2.cvtColor(anchored_bgr, cv2.COLOR_BGR2YUV)
        y_channel = image_yuv[:, :, 0].copy()

        y_modulated = self.modulator.modulate(y_channel, watermark_data)
        image_yuv[:, :, 0] = y_modulated

        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)