import cv2
import numpy as np
from typing import Tuple


class DCTModulator:
    """
    强鲁棒性 DCT 调制器
    使用低频系数嵌入，抗缩放、抗模糊、抗JPEG。
    """

    def __init__(self, strength: int = 50):
        self.strength = strength
        # === 关键修改：使用低频系数 ===
        # (0,0) 是直流分量(DC)，改动会导致整体亮度闪烁，不能动。
        # (0,1), (1,0) 是最低频交流分量，最抗缩放，但改动过大容易产生明显的方块效应。
        # (1,2), (2,1) 是次低频，是抗攻击和画质的最佳平衡点。
        self.c1_pos = (1, 2)
        self.c2_pos = (2, 1)

    def modulate(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        """嵌入逻辑"""
        img_float = image.astype(np.float32)
        h, w = img_float.shape
        data_len = len(watermark_data)

        # 遍历所有 8x8 块
        for y in range(0, h - 8 + 1, 8):
            for x in range(0, w - 8 + 1, 8):
                # 对角线索引，抗裁剪
                r, c = y // 8, x // 8
                bit_idx = (r + c) % data_len
                bit = watermark_data[bit_idx]

                # 提取块
                block = img_float[y:y + 8, x:x + 8]
                dct_block = cv2.dct(block)

                v1 = dct_block[self.c1_pos]
                v2 = dct_block[self.c2_pos]

                # 嵌入 Bit (使用加法调制，更稳定)
                # Bit 1: v1 必须比 v2 大 strength
                # Bit 0: v2 必须比 v1 大 strength
                if bit == 1:
                    if v1 <= v2 + self.strength:
                        diff = (v2 + self.strength - v1) / 2.0 + 1.0  # +1 为了容错
                        v1 += diff
                        v2 -= diff
                else:
                    if v2 <= v1 + self.strength:
                        diff = (v1 + self.strength - v2) / 2.0 + 1.0
                        v2 += diff
                        v1 -= diff

                dct_block[self.c1_pos] = v1
                dct_block[self.c2_pos] = v2

                # IDCT
                img_float[y:y + 8, x:x + 8] = cv2.idct(dct_block)

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def demodulate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        提取逻辑：返回软判决值（Soft Decision）
        """
        img_float = image.astype(np.float32)
        h, w = img_float.shape

        blocks_h = h // 8
        blocks_w = w // 8
        num_blocks = blocks_h * blocks_w

        if num_blocks == 0:
            return np.array([]), np.array([]), 0, 0

        raw_diffs = np.zeros(num_blocks, dtype=np.float32)

        idx = 0
        for y in range(0, blocks_h * 8, 8):
            for x in range(0, blocks_w * 8, 8):
                block = img_float[y:y + 8, x:x + 8]
                dct_block = cv2.dct(block)

                v1 = dct_block[self.c1_pos]
                v2 = dct_block[self.c2_pos]

                # 记录差值：正值代表1倾向，负值代表0倾向
                # 差值越大，置信度越高
                raw_diffs[idx] = v1 - v2
                idx += 1

        return raw_diffs, blocks_h, blocks_w