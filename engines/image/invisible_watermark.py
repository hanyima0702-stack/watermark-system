"""
暗水印主控制器

该模块实现暗水印系统的主控制器，协调嵌入和提取流程。
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .config import WatermarkConfig, EmbedResult, ExtractionResult
from .encoding.ecc_encoder import ECCEncoder
from .encoding.scrambler import Scrambler
from .embedding.macro_block import MacroBlockGenerator
from .embedding.ppm_modulator import PPMModulator
from .embedding.image_embedder import ImageEmbedder
from .extraction.fft_analyzer import FFTAnalyzer
from .extraction.geometric_corrector import GeometricCorrector
from .extraction.grid_aligner import GridAligner
from .extraction.majority_voter import MajorityVoter
from .utils.image_utils import calculate_psnr, calculate_ssim

logger = logging.getLogger(__name__)


class InvisibleWatermarkProcessor:
    """
    暗水印处理器
    
    协调所有模块，提供统一的嵌入和提取接口。
    """
    
    def __init__(self, config: Optional[WatermarkConfig] = None, 
                 config_path: Optional[str] = None):
        """
        初始化暗水印处理器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
            config_path: 配置文件路径，如果提供则从文件加载配置
        """
        # 加载配置
        if config_path is not None:
            self.config = WatermarkConfig.from_yaml(config_path)
            logger.info(f"Loaded configuration from: {config_path}")
        elif config is not None:
            self.config = config
        else:
            self.config = WatermarkConfig()
            logger.info("Using default configuration")
        
        # 验证配置
        self.config.validate()
        
        # 初始化编码模块
        self.ecc_encoder = ECCEncoder(
            code_type=self.config.ecc_type,
            n=self.config.ecc_n,
            k=self.config.ecc_k
        )
        self.scrambler = Scrambler(
            seed=self.config.scramble_seed,
            length=128
        )
        
        # 初始化嵌入模块
        self.block_generator = MacroBlockGenerator()
        self.ppm_modulator = PPMModulator(strength=self.config.modulation_strength)
        self.image_embedder = ImageEmbedder(
            block_generator=self.block_generator,
            modulator=self.ppm_modulator
        )
        
        # 初始化提取模块
        self.fft_analyzer = FFTAnalyzer(block_size=self.config.block_size)
        self.geometric_corrector = GeometricCorrector()
        self.grid_aligner = GridAligner(
            block_size=self.config.block_size,
            header_pattern=self.config.header_pattern
        )
        self.majority_voter = MajorityVoter(
            min_confidence=self.config.min_confidence
        )
        
        logger.info("InvisibleWatermarkProcessor initialized")
    
    def embed_watermark(self, image_path: str, watermark: str, 
                       output_path: str) -> EmbedResult:
        """
        嵌入水印
        
        协调编码、加扰、嵌入流程，将64位水印数据嵌入到图像中。
        
        Args:
            image_path: 原始图像路径
            watermark: 64位水印字符串（二进制字符串或整数）
            output_path: 输出路径
        
        Returns:
            嵌入结果对象
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting watermark embedding: {image_path}")
            
            # 1. 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            height, width = image.shape[:2]
            logger.info(f"Image size: {width}x{height}")
            
            # 2. 纠错编码
            logger.info("Encoding watermark data...")
            encoded_data = self.ecc_encoder.encode(watermark)
            logger.info(f"Encoded data length: {len(encoded_data)} bits")
            
            # 3. 加扰
            logger.info("Scrambling encoded data...")
            scrambled_data = self.scrambler.scramble(encoded_data)
            
            # 4. 嵌入水印
            logger.info("Embedding watermark into image...")
            watermarked_image = self.image_embedder.embed(image, scrambled_data)
            
            # 5. 保存图像
            cv2.imwrite(output_path, watermarked_image)
            logger.info(f"Watermarked image saved to: {output_path}")
            
            # 6. 计算质量指标
            psnr = calculate_psnr(image, watermarked_image)
            ssim = calculate_ssim(image, watermarked_image)
            
            # 7. 计算宏块数量
            block_count = self.image_embedder.calculate_block_count((height, width))
            
            processing_time = time.time() - start_time
            
            # 8. 生成嵌入报告
            result = EmbedResult(
                success=True,
                watermark_data=watermark if isinstance(watermark, str) else format(watermark, '064b'),
                encoded_data=''.join(str(b) for b in encoded_data),
                block_count=block_count,
                processing_time=processing_time,
                image_size=(height, width),
                quality_metrics={
                    'psnr': psnr,
                    'ssim': ssim
                }
            )
            
            logger.info(f"Embedding completed in {processing_time:.2f}s")
            logger.info(f"Quality metrics - PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}")
            logger.info(f"Block count: {block_count[0]}x{block_count[1]} = {block_count[0]*block_count[1]} blocks")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Embedding failed: {e}", exc_info=True)
            
            return EmbedResult(
                success=False,
                watermark_data="",
                encoded_data="",
                block_count=(0, 0),
                processing_time=processing_time,
                image_size=(0, 0),
                error_message=str(e)
            )

    def extract_watermark(self, image_path: str, 
                         visualize: bool = False) -> ExtractionResult:
        """
        提取水印
        
        协调FFT分析、几何校正、网格对齐、解调、投票、解码流程。
        实现180度旋转重试机制。
        
        Args:
            image_path: 带水印图像路径
            visualize: 是否生成可视化输出
        
        Returns:
            提取结果对象
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting watermark extraction: {image_path}")
            
            # 1. 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            height, width = image.shape[:2]
            logger.info(f"Image size: {width}x{height}")
            
            # 尝试提取（可能需要180度旋转重试）
            result = self._extract_with_retry(image, visualize)
            
            result.processing_time = time.time() - start_time
            logger.info(f"Extraction completed in {result.processing_time:.2f}s")
            
            if result.success:
                logger.info(f"Watermark extracted: {result.watermark_data}")
                logger.info(f"Confidence: {result.confidence:.3f}")
                logger.info(f"Valid blocks: {result.valid_blocks}/{result.total_blocks}")
            else:
                logger.warning(f"Extraction failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Extraction failed: {e}", exc_info=True)
            
            return ExtractionResult(
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _extract_with_retry(self, image: np.ndarray, 
                           visualize: bool) -> ExtractionResult:
        """
        尝试提取水印，如果失败则旋转180度重试
        
        Args:
            image: 输入图像
            visualize: 是否生成可视化输出
        
        Returns:
            提取结果对象
        """
        # 第一次尝试
        logger.info("First extraction attempt...")
        result = self._extract_internal(image, visualize)
        
        # 如果成功或不启用180度重试，直接返回
        if result.success or not self.config.enable_180_retry:
            return result
        
        # 如果Header验证失败，尝试180度旋转
        logger.info("First attempt failed, trying 180-degree rotation...")
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        result_rotated = self._extract_internal(rotated_image, visualize)
        
        # 返回置信度更高的结果
        if result_rotated.confidence > result.confidence:
            logger.info("180-degree rotation improved results")
            return result_rotated
        else:
            return result
    
    def _extract_internal(self, image: np.ndarray, 
                         visualize: bool) -> ExtractionResult:
        """
        内部提取方法
        
        Args:
            image: 输入图像
            visualize: 是否生成可视化输出
        
        Returns:
            提取结果对象
        """
        try:
            # 2. FFT分析 - 检测几何变换
            logger.info("Analyzing geometric transformations...")
            fft_result = self.fft_analyzer.analyze(image)
            rotation = fft_result['rotation']
            scale = fft_result['scale']
            fft_confidence = fft_result['confidence']
            
            logger.info(f"Detected rotation: {rotation:.2f}°, scale: {scale:.2f}, "
                       f"confidence: {fft_confidence:.3f}")
            
            # 3. 几何校正
            logger.info("Correcting geometric transformations...")
            corrected_image = self.geometric_corrector.correct(image, rotation, scale)
            
            # 转换为YUV色彩空间
            image_yuv = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2YUV)
            y_channel = image_yuv[:, :, 0]
            
            # 4. 网格对齐
            logger.info("Aligning grid...")
            x_offset, y_offset, align_confidence = self.grid_aligner.align(
                y_channel, self.ppm_modulator
            )
            
            logger.info(f"Grid aligned at offset ({x_offset}, {y_offset}), "
                       f"confidence: {align_confidence:.3f}")
            
            if align_confidence < 0.3:
                return ExtractionResult(
                    success=False,
                    confidence=align_confidence,
                    detected_rotation=rotation,
                    detected_scale=scale,
                    grid_offset=(x_offset, y_offset),
                    error_message="网格对齐置信度过低"
                )
            
            # 5. 从所有宏块中提取数据
            logger.info("Extracting data from blocks...")
            blocks_data, blocks_confidence = self._extract_from_blocks(
                y_channel, x_offset, y_offset
            )
            
            if len(blocks_data) == 0:
                return ExtractionResult(
                    success=False,
                    confidence=0.0,
                    detected_rotation=rotation,
                    detected_scale=scale,
                    grid_offset=(x_offset, y_offset),
                    total_blocks=0,
                    valid_blocks=0,
                    error_message="未找到有效的宏块"
                )
            
            logger.info(f"Extracted data from {len(blocks_data)} blocks")
            
            # 6. 多数投票
            logger.info("Performing majority voting...")
            voted_data, bit_confidences = self.majority_voter.vote(
                blocks_data, blocks_confidence
            )
            
            overall_confidence = float(np.mean(bit_confidences))
            logger.info(f"Voting confidence: {overall_confidence:.3f}")
            
            # 7. 解扰
            logger.info("Descrambling data...")
            descrambled_data = self.scrambler.descramble(voted_data)
            
            # 8. 纠错解码
            logger.info("Decoding watermark...")
            watermark_data, decode_success = self.ecc_encoder.decode(descrambled_data)
            
            if not decode_success or watermark_data is None:
                return ExtractionResult(
                    success=False,
                    confidence=overall_confidence,
                    detected_rotation=rotation,
                    detected_scale=scale,
                    grid_offset=(x_offset, y_offset),
                    total_blocks=len(blocks_data),
                    valid_blocks=len(blocks_data),
                    error_message="纠错解码失败"
                )
            
            # 9. 生成提取报告
            result = ExtractionResult(
                success=True,
                watermark_data=watermark_data,
                confidence=overall_confidence,
                bit_confidences=bit_confidences.tolist(),
                detected_rotation=rotation,
                detected_scale=scale,
                grid_offset=(x_offset, y_offset),
                total_blocks=len(blocks_data),
                valid_blocks=len(blocks_data),
                error_rate=1.0 - overall_confidence
            )
            
            # 10. 可视化（如果需要）
            if visualize and self.config.visualization_enabled:
                result.fft_spectrum = self.fft_analyzer.get_fft_spectrum(image)
                # 可以添加更多可视化数据
            
            return result
            
        except Exception as e:
            logger.error(f"Internal extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                success=False,
                error_message=str(e)
            )
    
    def _extract_from_blocks(self, y_channel: np.ndarray, 
                            x_offset: int, y_offset: int) \
                           -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        从所有宏块中提取数据
        
        Args:
            y_channel: Y通道图像
            x_offset: 网格x偏移量
            y_offset: 网格y偏移量
        
        Returns:
            (blocks_data, blocks_confidence) 宏块数据和置信度列表
        """
        height, width = y_channel.shape
        block_size = self.config.block_size
        
        blocks_data = []
        blocks_confidence = []
        
        # 获取宏块布局
        layout = self.block_generator.generate_block_layout()
        
        # 遍历所有宏块
        row = 0
        while y_offset + row * block_size + block_size <= height:
            col = 0
            while x_offset + col * block_size + block_size <= width:
                # 计算当前宏块的起始位置
                start_y = y_offset + row * block_size
                start_x = x_offset + col * block_size
                
                # 提取当前宏块
                block = y_channel[start_y:start_y + block_size,
                                 start_x:start_x + block_size]
                
                # 转换宏块内的相对位置为绝对位置
                payload_positions_abs = [
                    (start_y + x1, start_x + y1, start_y + x2, start_x + y2)
                    for x1, y1, x2, y2 in layout['payload']
                ]
                
                # 解调Payload
                try:
                    bits, confidences = self.ppm_modulator.demodulate(
                        y_channel, payload_positions_abs
                    )
                    
                    # 只保留置信度足够高的宏块
                    avg_confidence = np.mean(confidences)
                    if avg_confidence >= self.config.min_confidence:
                        blocks_data.append(bits)
                        blocks_confidence.append(confidences)
                
                except Exception as e:
                    logger.debug(f"Failed to extract from block at ({row}, {col}): {e}")
                
                col += 1
            row += 1
        
        return blocks_data, blocks_confidence
