# 明暗水印统一嵌入与识别系统

## 项目概述

政企单位文档、图片、音视频等内容的明暗水印统一嵌入与识别系统，支持全媒体内容的全生命周期可控、可识别、可追责。

## 系统架构

- **gateway/**: API网关和Web界面
- **business/**: 业务逻辑服务层
- **engines/**: 水印处理引擎
- **storage/**: 数据存储层
- **shared/**: 共享组件和工具

## 快速开始

```bash
# 启动开发环境
docker-compose up -d

# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/
```

## 技术栈

- **后端**: Python 3.9+, FastAPI, Celery
- **数据库**: PostgreSQL, Redis, ClickHouse
- **存储**: MinIO/OSS
- **容器**: Docker, Kubernetes
- **监控**: Prometheus, ELK Stack