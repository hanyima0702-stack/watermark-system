# fast_modulator.py
import numpy as np
from scipy.fft import dctn, idctn

class DCTModulator:
    """
    工业级极速 DCT 调制器 (全矩阵化/向量化实现)
    彻底消灭 Python for 循环
    """
    def __init__(self, strength: int = 100):
        self.strength = strength
        self.c1 = (1, 2)
        self.c2 = (2, 1)
        self.grid_h = 16
        self.grid_w = 8

    def modulate(self, image: np.ndarray, watermark_data: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32)
        h, w = img_float.shape

        # 1. 向下取整到 8 的倍数，防止边缘报错
        h_trunc = (h // 8) * 8
        w_trunc = (w // 8) * 8
        work_img = img_float[:h_trunc, :w_trunc]

        # 2. 图像重塑为 4D 块矩阵: (块行数, 块列数, 8, 8)
        # 这是向量化的核心魔法，一行代码完成图像切块
        blocks = work_img.reshape(h_trunc // 8, 8, w_trunc // 8, 8).transpose(0, 2, 1, 3)

        # 3. 批量执行二维 DCT (利用 scipy 底层 C 库)
        dct_blocks = dctn(blocks, axes=(2, 3), norm='ortho')

        # 4. 生成与图像等大的全屏水印二维网格
        wm_grid = watermark_data.reshape(self.grid_h, self.grid_w)
        repeats_y = int(np.ceil(dct_blocks.shape[0] / self.grid_h))
        repeats_x = int(np.ceil(dct_blocks.shape[1] / self.grid_w))
        full_wm = np.tile(wm_grid, (repeats_y, repeats_x))[:dct_blocks.shape[0], :dct_blocks.shape[1]]

        # 5. 提取特征系数并批量计算差值
        v1 = dct_blocks[:, :, self.c1[0], self.c1[1]]
        v2 = dct_blocks[:, :, self.c2[0], self.c2[1]]
        diff = np.zeros_like(v1)

        # 向量化判定条件: 比特位为 1
        mask_1 = (full_wm == 1) & (v1 <= v2 + self.strength)
        diff[mask_1] = (v2[mask_1] + self.strength - v1[mask_1]) / 2.0 + 1.0

        # 向量化判定条件: 比特位为 0
        mask_0 = (full_wm == 0) & (v2 <= v1 + self.strength)
        diff[mask_0] = (v1[mask_0] + self.strength - v2[mask_0]) / 2.0 + 1.0

        # 6. 批量施加修改
        dct_blocks[:, :, self.c1[0], self.c1[1]] += np.where(full_wm == 1, diff, -diff)
        dct_blocks[:, :, self.c2[0], self.c2[1]] -= np.where(full_wm == 1, diff, -diff)

        # 7. 批量执行逆 DCT 并拼回原图
        idct_blocks = idctn(dct_blocks, axes=(2, 3), norm='ortho')
        reconstructed = idct_blocks.transpose(0, 2, 1, 3).reshape(h_trunc, w_trunc)

        img_float[:h_trunc, :w_trunc] = reconstructed
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def demodulate(self, image: np.ndarray):
        """极速解调提取"""
        img_float = image.astype(np.float32)
        h, w = img_float.shape

        h_trunc = (h // 8) * 8
        w_trunc = (w // 8) * 8
        work_img = img_float[:h_trunc, :w_trunc]

        blocks = work_img.reshape(h_trunc // 8, 8, w_trunc // 8, 8).transpose(0, 2, 1, 3)
        dct_blocks = dctn(blocks, axes=(2, 3), norm='ortho')

        v1 = dct_blocks[:, :, self.c1[0], self.c1[1]]
        v2 = dct_blocks[:, :, self.c2[0], self.c2[1]]

        # 直接展平返回所有块的差值
        raw_diffs = (v1 - v2).flatten()
        return raw_diffs, h_trunc // 8, w_trunc // 8