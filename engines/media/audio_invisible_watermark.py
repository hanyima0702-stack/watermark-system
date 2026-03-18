"""
音频暗水印处理器
实现回声隐藏(Echo Hiding)和基于FFT的频域相位编码水印
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import io
import logging
from pydub import AudioSegment
import numpy as np
from pathlib import Path
from scipy import signal
import warnings
import numpy as np
from collections import Counter


logger = logging.getLogger(__name__)


class AudioInvisibleWatermarkConfig:
    """音频暗水印配置"""
    
    def __init__(
        self,
        method: str = 'echo_hiding',
        strength: float = 1.0,
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
            delay_0: int = 150,
            delay_1: int = 250,
            decay: float = 0.5,  # 恢复到0.5，避免回声过大干扰音质
            strength: float = 0.5
    ):
        self.delay_0 = delay_0
        self.delay_1 = delay_1
        self.decay = decay
        self.strength = strength
        self.segment_length = 4096

        # 【修复2】16位超强同步头，自然界噪音几乎不可能撞上
        self.sync_header = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=int)

    def _compute_cepstrum(self, signal: np.ndarray) -> np.ndarray:
        windowed_signal = signal * np.hanning(len(signal))
        spectrum = np.fft.fft(windowed_signal)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.real(np.fft.ifft(log_spectrum))
        return np.abs(cepstrum[:len(cepstrum) // 2])

    def embed(self, audio: np.ndarray, watermark_bits: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        if len(audio.shape) > 1:
            watermarked = audio.copy()
            watermarked[:, 0] = self._embed_mono(audio[:, 0], watermark_bits)
            return watermarked
        else:
            return self._embed_mono(audio, watermark_bits)

    def _embed_mono(self, audio: np.ndarray, watermark_bits: np.ndarray) -> np.ndarray:
        frame_bits = np.concatenate((self.sync_header, watermark_bits))
        frame_len = len(frame_bits)
        total_segments = len(audio) // self.segment_length

        if total_segments < frame_len:
            raise ValueError(f"Audio is too short! Need at least {frame_len * self.segment_length} samples.")

        repeated_bits = np.tile(frame_bits, total_segments // frame_len + 1)[:total_segments]
        watermarked = audio.copy().astype(np.float64)

        for i, bit in enumerate(repeated_bits):
            start_idx = i * self.segment_length
            end_idx = start_idx + self.segment_length
            segment = audio[start_idx:end_idx]

            delay = self.delay_1 if bit == 1 else self.delay_0
            echo = np.zeros_like(segment)
            if len(segment) > delay:
                echo[delay:] = segment[:-delay] * self.decay * self.strength

            watermarked[start_idx:end_idx] = segment + echo

        max_val = np.abs(watermarked).max()
        if max_val > 1.0:
            watermarked = watermarked / max_val * 0.95

        return watermarked

    def extract(self, watermarked_audio: np.ndarray, watermark_length: int, sample_rate: int = 44100):
        if len(watermarked_audio.shape) > 1:
            audio = watermarked_audio[:, 0]
        else:
            audio = watermarked_audio

        header_len = len(self.sync_header)
        best_payloads = []
        max_score = -1
        step_size = 512

        for offset in range(0, self.segment_length, step_size):
            shifted_audio = audio[offset:]
            total_segments = len(shifted_audio) // self.segment_length

            if total_segments < header_len + watermark_length:
                continue

            raw_bits = np.zeros(total_segments, dtype=int)

            for i in range(total_segments):
                start_idx = i * self.segment_length
                end_idx = start_idx + self.segment_length
                segment = shifted_audio[start_idx:end_idx]
                cepstrum = self._compute_cepstrum(segment)

                # 【修复1】局部基线减法：计算周围 10 个点的平均能量作为基线
                # 同时允许峰值有 1 个采样点的偏移容错
                if self.delay_1 + 6 > len(cepstrum):
                    continue

                peak_0 = np.max(cepstrum[self.delay_0 - 1: self.delay_0 + 2])
                base_0 = np.mean(cepstrum[self.delay_0 - 5: self.delay_0 + 6])
                prominence_0 = peak_0 - base_0  # 相对凸起高度

                peak_1 = np.max(cepstrum[self.delay_1 - 1: self.delay_1 + 2])
                base_1 = np.mean(cepstrum[self.delay_1 - 5: self.delay_1 + 6])
                prominence_1 = peak_1 - base_1

                raw_bits[i] = 1 if prominence_1 > prominence_0 else 0

            found_payloads = []

            # 【修复3】必须零误码 (err == 0)，拒绝所有假密码
            for i in range(len(raw_bits) - header_len - watermark_length + 1):
                window = raw_bits[i:i + header_len]
                err = np.sum(np.abs(window - self.sync_header))

                if err == 0:
                    payload_start = i + header_len
                    payload_end = payload_start + watermark_length
                    found_payloads.append(raw_bits[payload_start:payload_end])

            score = len(found_payloads)
            if score > max_score and score > 0:
                max_score = score
                best_payloads = found_payloads

        if not best_payloads:
            logger.warning("Failed to find any matching sync headers.")
            return np.zeros(watermark_length, dtype=int), 0.0

        final_bits = np.zeros(watermark_length, dtype=int)
        for bit_idx in range(watermark_length):
            column_bits = [payload[bit_idx] for payload in best_payloads]
            final_bits[bit_idx] = Counter(column_bits).most_common(1)[0][0]

        confidence = min(1.0, float(len(best_payloads)) / 3.0)
        return final_bits, confidence


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

    def _read_audio(self, file_path: Union[str, Path]) -> Tuple[int, np.ndarray]:
        """通用的音频读取方法，支持各种格式"""
        # pydub 会自动根据文件头识别格式（MP3, FLAC, M4A, WAV等）
        audio_seg = AudioSegment.from_file(str(file_path))
        sample_rate = audio_seg.frame_rate

        # 转换为 numpy 数组
        samples = np.array(audio_seg.get_array_of_samples())

        # 处理立体声
        if audio_seg.channels == 2:
            samples = samples.reshape((-1, 2))

        # 归一化到 [-1.0, 1.0] 的浮点数
        # sample_width 是字节数，比如 16-bit 音频的 width 是 2
        max_val = float(1 << (8 * audio_seg.sample_width - 1))
        audio_np = samples.astype(np.float32) / max_val

        return sample_rate, audio_np

    def _write_audio(self, audio_np: np.ndarray, sample_rate: int, file_path: Union[str, Path]):
        """通用的音频写入方法，根据后缀名自动导出格式"""
        # 将归一化的浮点数转换回 16-bit 整数
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        channels = 1 if len(audio_int16.shape) == 1 else 2

        # 将 numpy 数组塞回 pydub 的 AudioSegment
        watermarked_seg = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 固定导出为 16-bit 深度
            channels=channels
        )

        # 获取目标文件的后缀名（比如 'mp3', 'flac'），去掉点号
        ext = Path(file_path).suffix.lower().lstrip('.')
        if not ext:
            ext = 'wav'

        # 导出文件
        # 注意：导出 mp3 等有损格式可能会削弱水印信号
        watermarked_seg.export(str(file_path), format=ext)

    def embed_file(
            self,
            input_path: Union[str, Path],
            watermark_data: str
    ) -> bytes:
        """
        在音频文件中嵌入暗水印，返回含水印的音频 bytes，不涉及存储。

        Args:
            input_path:     原始音频路径
            watermark_data: 水印比特字符串（如 "10110..."）

        Returns:
            含水印的音频 bytes
        """
        sample_rate, audio = self._read_audio(input_path)

        clean_bits = [int(b) for b in watermark_data if b in ('0', '1')]
        watermark_bits = np.array(clean_bits, dtype=np.uint8)
        if len(watermark_bits) == 0:
            raise ValueError("Watermark data must contain 0s and 1s.")

        watermarked = self.processor.embed(audio, watermark_bits, sample_rate)

        # 序列化为 bytes（使用原始格式）
        audio_int16 = np.clip(watermarked * 32767, -32768, 32767).astype(np.int16)
        channels = 1 if len(audio_int16.shape) == 1 else 2
        from pydub import AudioSegment
        seg = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=channels,
        )
        ext = Path(input_path).suffix.lower().lstrip('.') or 'wav'
        buf = io.BytesIO()
        seg.export(buf, format=ext)
        logger.info(f"音频暗水印嵌入完毕，{len(watermark_bits)} bits")
        return buf.getvalue()

    async def process_watermark(
            self,
            input_path: Union[str, Path],
            minio_service,
            invisible_watermark: str = None,
            object_key: str = None,
            bucket_name: str = None,
    ) -> dict:
        """
        上层方法：嵌入暗水印后上传 MinIO。
        （音频暂无明水印，预留扩展）

        Args:
            input_path:          原始音频路径
            minio_service:       MinIOService 实例
            invisible_watermark: 暗水印比特字符串，None 则直接上传原文件
            object_key:          MinIO 对象键，不传则自动生成
            bucket_name:         目标 bucket，不传则使用 result_bucket

        Returns:
            dict，包含 success、minio_object_key 等字段
        """
        import time, uuid
        start_time = time.time()
        try:
            ext = Path(input_path).suffix.lower().lstrip('.') or 'wav'

            if invisible_watermark:
                audio_bytes = self.embed_file(input_path, invisible_watermark)
            else:
                with open(input_path, 'rb') as f:
                    audio_bytes = f.read()

            target_bucket = bucket_name or minio_service.config.result_bucket
            target_key = object_key or f"watermarked/{uuid.uuid4().hex}.{ext}"

            await minio_service.upload_file(
                bucket_name=target_bucket,
                object_key=target_key,
                file_data=audio_bytes,
                content_type=f"audio/{ext}",
                metadata={"watermark": invisible_watermark or ""},
            )

            return {
                "success": True,
                "minio_object_key": f"{target_bucket}/{target_key}",
                "processing_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"音频水印处理失败: {e}")
            return {"success": False, "error": str(e)}

    def extract_file(
            self,
            input_path: Union[str, Path],
            watermark_length: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """从音频文件中提取水印（纯比特流模式）"""
        try:
            # 1. 读取音频
            sample_rate, audio = self._read_audio(input_path)

            # 如果未指定长度，使用默认比特数
            if watermark_length is None:
                watermark_length = 64

                # 2. 提取水印比特
            extracted_bits, confidence_array = self.processor.extract(
                audio, watermark_length, sample_rate
            )

            # 3. 【核心修改】将 [1, 0, 1, 0] 数组直接拼回 "1010" 字符串，跳过 ASCII 转换
            extracted_string = "".join(str(int(bit)) for bit in extracted_bits)
            avg_confidence = np.mean(confidence_array)

            logger.info(f"Extracted watermark with confidence {avg_confidence:.2f}")
            return extracted_string, float(avg_confidence)

        except Exception as e:
            logger.error(f"Failed to extract watermark: {e}")
            return None, 0.0
    


    
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
            # 使用新的通用读取方法
            sr1, audio1 = self._read_audio(original_path)
            sr2, audio2 = self._read_audio(watermarked_path)
            
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
