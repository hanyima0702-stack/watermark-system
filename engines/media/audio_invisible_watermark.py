"""
音频暗水印处理器
实现回声隐藏(Echo Hiding)和基于FFT的频域相位编码水印
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from scipy.io import wavfile
from scipy import signal
import warnings

logger = logging.getLogger(__name__)


class AudioInvisibleWatermarkConfig:
    """音频暗水印配置"""
    
    def __init__(
        self,
        method: str = 'echo_hiding',
        strength: float = 0.5,
        echo_delay: int = 100,
        echo_decay: float = 0.5,
        sample_rate: int = 44100,
        frame_size: int = 2048,
        hop_size: int = 512
    ):
        """
        Args:
            method: 水印方法 ('echo_hiding', 'phase_coding')
            strength: 水印强度 (0.0-1.0)
            echo_delay: 回声延迟（样本数）
            echo_decay: 回声衰减系数
            sample_rate: 采样率
            frame_size: FFT帧大小
            hop_size: 帧跳跃大小
        """
        self.method = method
        self.strength = strength
        self.echo_delay = echo_delay
        self.echo_decay = echo_decay
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size


class EchoHidingWatermark:
    """回声隐藏音频水印算法"""
    
    def __init__(
        self,
        delay_0: int = 100,
        delay_1: int = 150,
        decay: float = 0.5,
        strength: float = 0.5
    ):
        """
        Args:
            delay_0: 表示比特0的回声延迟（样本数）
            delay_1: 表示比特1的回声延迟（样本数）
            decay: 回声衰减系数
            strength: 水印强度
        """
        self.delay_0 = delay_0
        self.delay_1 = delay_1
        self.decay = decay
        self.strength = strength
    
    def embed(
        self,
        audio: np.ndarray,
        watermark_bits: np.ndarray,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        使用回声隐藏技术嵌入水印
        
        Args:
            audio: 音频信号 (单声道或立体声)
            watermark_bits: 水印比特序列
            sample_rate: 采样率
            
        Returns:
            含水印的音频信号
        """
        # 处理立体声
        if len(audio.shape) > 1:
            # 只在第一个声道嵌入
            watermarked = audio.copy()
            watermarked[:, 0] = self._embed_mono(audio[:, 0], watermark_bits, sample_rate)
            return watermarked
        else:
            return self._embed_mono(audio, watermark_bits, sample_rate)
    
    def _embed_mono(
        self,
        audio: np.ndarray,
        watermark_bits: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """在单声道音频中嵌入水印"""
        # 计算每个比特的段长度
        segment_length = len(audio) // len(watermark_bits)
        
        if segment_length < max(self.delay_0, self.delay_1) * 2:
            raise ValueError(
                f"Audio too short for {len(watermark_bits)} bits. "
                f"Need at least {max(self.delay_0, self.delay_1) * 2 * len(watermark_bits)} samples"
            )
        
        watermarked = audio.copy().astype(np.float64)
        
        # 对每个比特段嵌入回声
        for i, bit in enumerate(watermark_bits):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(audio))
            segment = audio[start_idx:end_idx]
            
            # 根据比特值选择延迟
            delay = self.delay_1 if bit == 1 else self.delay_0
            
            # 创建回声信号
            echo = np.zeros_like(segment)
            if len(segment) > delay:
                echo[delay:] = segment[:-delay] * self.decay * self.strength
            
            # 混合原始信号和回声
            watermarked[start_idx:end_idx] = segment + echo
        
        # 归一化防止溢出
        max_val = np.abs(watermarked).max()
        if max_val > 1.0:
            watermarked = watermarked / max_val * 0.95
        
        return watermarked
    
    def extract(
        self,
        watermarked_audio: np.ndarray,
        watermark_length: int,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从音频中提取水印
        
        Args:
            watermarked_audio: 含水印的音频信号
            watermark_length: 水印比特数
            sample_rate: 采样率
            
        Returns:
            (extracted_bits, confidence): 提取的比特序列和置信度
        """
        # 处理立体声
        if len(watermarked_audio.shape) > 1:
            audio = watermarked_audio[:, 0]
        else:
            audio = watermarked_audio
        
        segment_length = len(audio) // watermark_length
        extracted_bits = np.zeros(watermark_length, dtype=int)
        confidence = np.zeros(watermark_length)
        
        # 对每个段进行自相关分析
        for i in range(watermark_length):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, len(audio))
            segment = audio[start_idx:end_idx]

            cepstrum = self._compute_cepstrum(segment)

            # 检测两个延迟位置的峰值
            peak_0 = cepstrum[self.delay_0] if self.delay_0 < len(cepstrum) else 0
            peak_1 = cepstrum[self.delay_1] if self.delay_1 < len(cepstrum) else 0

            # 根据峰值大小判断比特
            total_peak = abs(peak_0) + abs(peak_1) + 1e-10
            if peak_1 > peak_0:
                extracted_bits[i] = 1
                confidence[i] = min(abs(peak_1) / total_peak, 1.0)
            else:
                extracted_bits[i] = 0
                confidence[i] = min(abs(peak_0) / total_peak, 1.0)
        
        return extracted_bits, confidence
    
    def _compute_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """计算信号的自相关"""
        n = len(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:]  # 只取正延迟部分
        autocorr = autocorr / autocorr[0]  # 归一化
        return autocorr

    def _compute_cepstrum(self, signal: np.ndarray) -> np.ndarray:
        """计算信号的实倒谱 (Real Cepstrum)"""
        # 加窗，减少 FFT 的频谱泄漏
        windowed_signal = signal * np.hanning(len(signal))

        # 1. 计算 FFT
        spectrum = np.fft.fft(windowed_signal)

        # 2. 计算幅度谱的对数 (加一个小偏置防止 log(0))
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)

        # 3. 计算 IFFT 得到倒谱
        cepstrum = np.real(np.fft.ifft(log_spectrum))

        # 只需要正延迟部分，并且取绝对值以便寻找峰值
        half_len = len(cepstrum) // 2
        return np.abs(cepstrum[:half_len])


class PhaseCodingWatermark:
    """基于FFT的频域相位编码水印"""
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: int = 512,
        phase_shift: float = np.pi / 4,
        strength: float = 0.5
    ):
        """
        Args:
            frame_size: FFT帧大小
            hop_size: 帧跳跃大小
            phase_shift: 相位偏移量
            strength: 水印强度
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.phase_shift = phase_shift
        self.strength = strength
    
    def embed(
        self,
        audio: np.ndarray,
        watermark_bits: np.ndarray,
        sample_rate: int = 44100
    ) -> np.ndarray:
        """
        使用相位编码嵌入水印
        
        Args:
            audio: 音频信号
            watermark_bits: 水印比特序列
            sample_rate: 采样率
            
        Returns:
            含水印的音频信号
        """
        # 处理立体声
        if len(audio.shape) > 1:
            watermarked = audio.copy()
            watermarked[:, 0] = self._embed_mono(audio[:, 0], watermark_bits)
            return watermarked
        else:
            return self._embed_mono(audio, watermark_bits)
    
    def _embed_mono(
        self,
        audio: np.ndarray,
        watermark_bits: np.ndarray
    ) -> np.ndarray:
        """在单声道音频中嵌入水印"""
        # 分帧
        frames = self._frame_signal(audio, self.frame_size, self.hop_size)
        num_frames = frames.shape[0]
        
        # 计算每个比特使用多少帧
        frames_per_bit = max(1, num_frames // len(watermark_bits))
        
        watermarked_frames = frames.copy()
        
        # 对每个比特对应的帧进行相位编码
        for i, bit in enumerate(watermark_bits):
            start_frame = i * frames_per_bit
            end_frame = min((i + 1) * frames_per_bit, num_frames)
            
            for frame_idx in range(start_frame, end_frame):
                frame = frames[frame_idx]
                
                # FFT变换
                fft_frame = np.fft.fft(frame)
                magnitude = np.abs(fft_frame)
                phase = np.angle(fft_frame)
                
                # 在中频区域修改相位
                freq_start = self.frame_size // 8
                freq_end = self.frame_size // 4
                
                # 根据比特值调整相位
                phase_delta = self.phase_shift * self.strength if bit == 1 else -self.phase_shift * self.strength
                phase[freq_start:freq_end] += phase_delta
                
                # 重构信号
                fft_modified = magnitude * np.exp(1j * phase)
                watermarked_frames[frame_idx] = np.real(np.fft.ifft(fft_modified))
        
        # 重叠相加重构信号
        watermarked = self._overlap_add(watermarked_frames, self.hop_size, len(audio))
        
        # 归一化
        max_val = np.abs(watermarked).max()
        if max_val > 1.0:
            watermarked = watermarked / max_val * 0.95
        
        return watermarked
    
    def extract(
        self,
        watermarked_audio: np.ndarray,
        watermark_length: int,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从音频中提取水印
        
        Args:
            watermarked_audio: 含水印的音频信号
            watermark_length: 水印比特数
            sample_rate: 采样率
            
        Returns:
            (extracted_bits, confidence): 提取的比特序列和置信度
        """
        # 处理立体声
        if len(watermarked_audio.shape) > 1:
            audio = watermarked_audio[:, 0]
        else:
            audio = watermarked_audio
        
        # 分帧
        frames = self._frame_signal(audio, self.frame_size, self.hop_size)
        num_frames = frames.shape[0]
        
        frames_per_bit = max(1, num_frames // watermark_length)
        
        extracted_bits = np.zeros(watermark_length, dtype=int)
        confidence = np.zeros(watermark_length)
        
        # 对每个比特对应的帧进行相位分析
        for i in range(watermark_length):
            start_frame = i * frames_per_bit
            end_frame = min((i + 1) * frames_per_bit, num_frames)
            
            phase_sum = 0
            frame_count = 0
            
            for frame_idx in range(start_frame, end_frame):
                frame = frames[frame_idx]
                
                # FFT变换
                fft_frame = np.fft.fft(frame)
                phase = np.angle(fft_frame)
                
                # 分析中频区域的相位
                freq_start = self.frame_size // 8
                freq_end = self.frame_size // 4
                
                avg_phase = np.mean(phase[freq_start:freq_end])
                phase_sum += avg_phase
                frame_count += 1
            
            avg_phase = phase_sum / frame_count if frame_count > 0 else 0
            
            # 根据平均相位判断比特
            if avg_phase > 0:
                extracted_bits[i] = 1
                confidence[i] = min(abs(avg_phase) / self.phase_shift, 1.0)
            else:
                extracted_bits[i] = 0
                confidence[i] = min(abs(avg_phase) / self.phase_shift, 1.0)
        
        return extracted_bits, confidence
    
    def _frame_signal(
        self,
        signal: np.ndarray,
        frame_size: int,
        hop_size: int
    ) -> np.ndarray:
        """将信号分帧"""
        num_frames = (len(signal) - frame_size) // hop_size + 1
        frames = np.zeros((num_frames, frame_size))
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            if end <= len(signal):
                frames[i] = signal[start:end]
        
        return frames
    
    def _overlap_add(
        self,
        frames: np.ndarray,
        hop_size: int,
        signal_length: int
    ) -> np.ndarray:
        """重叠相加重构信号"""
        num_frames, frame_size = frames.shape
        signal = np.zeros(signal_length)
        window = np.hanning(frame_size)
        
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_size
            if end <= signal_length:
                signal[start:end] += frames[i] * window
        
        return signal


class AudioInvisibleWatermarker:
    """音频暗水印处理器主类"""
    
    def __init__(self, config: Optional[AudioInvisibleWatermarkConfig] = None):
        """
        Args:
            config: 音频暗水印配置
        """
        self.config = config or AudioInvisibleWatermarkConfig()
        
        # 初始化处理器
        if self.config.method == 'echo_hiding':
            self.processor = EchoHidingWatermark(
                delay_0=self.config.echo_delay,
                delay_1=int(self.config.echo_delay * 1.5),
                decay=self.config.echo_decay,
                strength=self.config.strength
            )
        elif self.config.method == 'phase_coding':
            self.processor = PhaseCodingWatermark(
                frame_size=self.config.frame_size,
                hop_size=self.config.hop_size,
                phase_shift=np.pi / 4,
                strength=self.config.strength
            )
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
    
    def embed_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        watermark_data: str
    ) -> bool:
        """
        在音频文件中嵌入水印
        
        Args:
            input_path: 输入音频文件路径
            output_path: 输出音频文件路径
            watermark_data: 水印数据（字符串）
            
        Returns:
            是否成功
        """
        try:
            # 读取音频文件
            sample_rate, audio = wavfile.read(str(input_path))
            
            # 归一化到[-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # 转换水印数据为比特
            watermark_bits = np.array(list(watermark_data), dtype=np.uint8)
            
            # 嵌入水印
            watermarked = self.processor.embed(audio, watermark_bits, sample_rate)
            
            # 转换回整数格式
            watermarked_int = (watermarked * 32767).astype(np.int16)
            
            # 保存
            wavfile.write(str(output_path), sample_rate, watermarked_int)
            
            logger.info(f"Successfully embedded watermark in {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed watermark: {e}")
            return False
    
    def extract_file(
        self,
        input_path: Union[str, Path],
        watermark_length: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """
        从音频文件中提取水印
        
        Args:
            input_path: 输入音频文件路径
            watermark_length: 水印长度（字符数），如果为None则尝试自动检测
            
        Returns:
            (watermark_data, confidence): 提取的水印数据和平均置信度
        """
        try:
            # 读取音频文件
            sample_rate, audio = wavfile.read(str(input_path))
            
            # 归一化
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            
            # 如果未指定长度，使用默认值
            if watermark_length is None:
                watermark_length = 64  # 默认16个字符 = 128比特
            

            # 提取水印
            extracted_bits, confidence_array = self.processor.extract(
                audio, watermark_length, sample_rate
            )
            

            avg_confidence = np.mean(confidence_array)
            
            logger.info(f"Extracted watermark with confidence {avg_confidence:.2f}")
            return extracted_bits, float(avg_confidence)
            
        except Exception as e:
            logger.error(f"Failed to extract watermark: {e}")
            return None, 0.0
    
    def _string_to_bits(self, text: str) -> np.ndarray:
        """将字符串转换为比特序列"""
        bits = []
        for char in text:
            byte_val = ord(char)
            for i in range(8):
                bits.append((byte_val >> (7 - i)) & 1)
        return np.array(bits, dtype=int)
    
    def _bits_to_string(self, bits: np.ndarray) -> str:
        """将比特序列转换为字符串"""
        chars = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_val = 0
                for j, bit in enumerate(byte_bits):
                    byte_val |= (int(bit) << (7 - j))
                
                # 只添加可打印字符
                if 32 <= byte_val <= 126:
                    chars.append(chr(byte_val))
                else:
                    chars.append('?')
        
        return ''.join(chars)
    
    def measure_quality(
        self,
        original_path: Union[str, Path],
        watermarked_path: Union[str, Path]
    ) -> dict:
        """
        测量音频质量
        
        Args:
            original_path: 原始音频路径
            watermarked_path: 含水印音频路径
            
        Returns:
            质量指标字典
        """
        try:
            # 读取音频
            sr1, audio1 = wavfile.read(str(original_path))
            sr2, audio2 = wavfile.read(str(watermarked_path))
            
            if sr1 != sr2:
                raise ValueError("Sample rates don't match")
            
            # 归一化
            if audio1.dtype == np.int16:
                audio1 = audio1.astype(np.float32) / 32768.0
            if audio2.dtype == np.int16:
                audio2 = audio2.astype(np.float32) / 32768.0
            
            # 处理立体声
            if len(audio1.shape) > 1:
                audio1 = audio1[:, 0]
            if len(audio2.shape) > 1:
                audio2 = audio2[:, 0]
            
            # 确保长度相同
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # 计算SNR (Signal-to-Noise Ratio)
            noise = audio2 - audio1
            signal_power = np.mean(audio1 ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # 计算PSNR (Peak Signal-to-Noise Ratio)
            mse = np.mean((audio1 - audio2) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            
            # 计算相关系数
            correlation = np.corrcoef(audio1, audio2)[0, 1]
            
            return {
                'snr_db': float(snr),
                'psnr_db': float(psnr),
                'correlation': float(correlation),
                'imperceptible': snr > 20  # SNR > 20dB 通常认为不可察觉
            }
            
        except Exception as e:
            logger.error(f"Failed to measure quality: {e}")
            return {}
