import cv2
import numpy as np
from typing import Tuple


class DCTModulator:
    """
    强鲁棒性 DCT 调制器 (二维分块映射版)
    """

    def __init__(self, strength: int = 50):
        self.strength = strength
        self.c1_pos = (1, 2)
        self.c2_pos = (2, 1)

        # 定义二维网格尺寸 (16 * 8 = 128 bit)
        self.grid_h = 16
        self.grid_w = 8

    def modulate(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32)
        h, w = img_float.shape
        data_len = len(watermark_data)

        if data_len != self.grid_h * self.grid_w:
            raise ValueError(f"水印数据长度必须为 {self.grid_h * self.grid_w}")

        for y in range(0, h - 8 + 1, 8):
            for x in range(0, w - 8 + 1, 8):
                r, c = y // 8, x // 8

                # === 核心改进：二维网格映射 ===
                block_r = r % self.grid_h
                block_c = c % self.grid_w
                bit_idx = block_r * self.grid_w + block_c

                bit = watermark_data[bit_idx]

                block = img_float[y:y + 8, x:x + 8]
                dct_block = cv2.dct(block)

                v1 = dct_block[self.c1_pos]
                v2 = dct_block[self.c2_pos]

                if bit == 1:
                    if v1 <= v2 + self.strength:
                        diff = (v2 + self.strength - v1) / 2.0 + 1.0
                        v1 += diff
                        v2 -= diff
                else:
                    if v2 <= v1 + self.strength:
                        diff = (v1 + self.strength - v2) / 2.0 + 1.0
                        v2 += diff
                        v1 -= diff

                dct_block[self.c1_pos] = v1
                dct_block[self.c2_pos] = v2

                img_float[y:y + 8, x:x + 8] = cv2.idct(dct_block)

        return np.clip(img_float, 0, 255).astype(np.uint8)

    def demodulate(self, image: np.ndarray):
        # demodulate 逻辑无需改变，它只负责提取每一个 8x8 块的原始差值
        # 具体的二维重组逻辑在 Processor 中处理，以保持类的职责单一
        # ... (保留原有的 demodulate 代码) ...
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

                raw_diffs[idx] = v1 - v2
                idx += 1

        return raw_diffs, blocks_h, blocks_w