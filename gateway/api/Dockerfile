# API Gateway Dockerfile
FROM watermark-system-base:latest

# 复制API Gateway代码
COPY gateway/api/ ./gateway/api/
COPY business/ ./business/
COPY storage/ ./storage/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "gateway.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]