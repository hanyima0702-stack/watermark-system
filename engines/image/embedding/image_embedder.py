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
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # ==========================================
        # 通道 1: 在 Y (亮度) 通道嵌入 DCT 水印
        # ==========================================
        y_channel = image_yuv[:, :, 0].copy()
        y_modulated = self.modulator.modulate(y_channel, watermark_data)
        image_yuv[:, :, 0] = y_modulated

        # ==========================================
        # 通道 2: 在 U (色度) 通道铺设空间锚点
        # ==========================================
        if cfg.enable_spatial_anchors and cfg.anchor_strength > 0:
            h, w = image.shape[:2]
            anchor_patch = self._generate_anchor_pattern(cfg.anchor_spacing)
            noise_layer = np.zeros((h, w), dtype=np.float32)

            for y in range(0, h, cfg.anchor_spacing):
                for x in range(0, w, cfg.anchor_spacing):
                    patch_h = min(cfg.anchor_spacing, h - y)
                    patch_w = min(cfg.anchor_spacing, w - x)
                    noise_layer[y:y + patch_h, x:x + patch_w] = anchor_patch[:patch_h, :patch_w]

            # 提取 U 通道，叠加锚点，再放回去
            u_float = image_yuv[:, :, 1].astype(np.float32)
            u_float += noise_layer * cfg.anchor_strength
            image_yuv[:, :, 1] = np.clip(u_float, 0, 255).astype(np.uint8)

        # 合并通道转回 BGR
        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)