# Media Processing Engine Dockerfile
FROM watermark-system-base:latest

# 安装额外的音视频处理依赖
RUN apt-get update && apt-get install -y \
    # FFmpeg 完整版本
    ffmpeg \
    # 音频处理库
    libsndfile1-dev \
    libfftw3-dev \
    # 视频编解码器
    libx264-dev \
    libx265-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制媒体处理引擎代码
COPY engines/media/ ./engines/media/
COPY storage/ ./storage/

# 启动媒体处理服务
CMD ["python", "-m", "engines.media.main"]