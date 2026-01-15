# Image Processing Engine Dockerfile
FROM watermark-system-base:latest

# 安装额外的图像处理依赖
RUN apt-get update && apt-get install -y \
    # 高级图像处理库
    libmagickwand-dev \
    imagemagick \
    # 字体支持
    fonts-noto \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 复制图像处理引擎代码
COPY engines/image/ ./engines/image/
COPY storage/ ./storage/

# 启动图像处理服务
CMD ["python", "-m", "engines.image.main"]