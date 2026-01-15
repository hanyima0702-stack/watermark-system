# Document Processing Engine Dockerfile
FROM watermark-system-base:latest

# 安装额外的文档处理依赖
RUN apt-get update && apt-get install -y \
    # Microsoft Office 兼容性
    fonts-liberation \
    fonts-dejavu-core \
    # PDF处理增强
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 复制文档处理引擎代码
COPY engines/document/ ./engines/document/
COPY storage/ ./storage/

# 启动文档处理服务
CMD ["python", "-m", "engines.document.main"]