# API Gateway and Web Interface

明暗水印统一嵌入与识别系统 - API网关和Web用户界面

## 概述

本模块实现了系统的API网关和Web用户界面，提供统一的访问入口和用户交互界面。

## 功能特性

### API网关 (gateway/api/)

1. **统一API入口**
   - RESTful API设计
   - 请求路由和负载均衡
   - API版本管理 (v1, v2)
   - 向后兼容性支持

2. **限流和监控**
   - 基于Redis的分布式限流
   - Prometheus指标收集
   - 请求日志记录
   - 性能监控

3. **认证和授权**
   - JWT令牌认证
   - 基于角色的访问控制 (RBAC)
   - 多因素认证支持
   - 会话管理

4. **文件服务**
   - 大文件分片上传
   - 断点续传支持
   - 文件类型验证
   - 安全扫描
   - 临时下载链接

### Web用户界面 (watermark-frontend/)

1. **文件上传**
   - 拖拽上传
   - 分片上传
   - 进度显示
   - 文件验证

2. **水印配置向导**
   - 可视化配置界面
   - 明水印配置
   - 暗水印配置
   - 实时预览

3. **任务管理**
   - 任务状态跟踪
   - 进度显示
   - 结果下载
   - 任务取消

## 目录结构

```
gateway/
├── api/                    # API网关服务
│   ├── main.py            # 主应用入口
│   ├── config.py          # 配置管理
│   ├── middleware.py      # 中间件（限流、日志、认证、监控）
│   ├── health.py          # 健康检查端点
│   ├── file_service.py    # 文件上传下载服务
│   ├── models.py          # API模型定义
│   ├── dependencies.py    # 依赖注入
│   └── routers/           # API路由
│       ├── v1.py          # API v1
│       └── v2.py          # API v2
└── web/                   # Web界面服务

watermark-frontend/
├── src/
│   ├── components/        # React组件
│   │   ├── FileUpload.tsx              # 文件上传组件
│   │   ├── WatermarkConfigWizard.tsx   # 配置向导
│   │   ├── TaskProgress.tsx            # 任务进度
│   │   └── __tests__/                  # 组件测试
│   ├── pages/             # 页面组件
│   │   ├── Login.tsx
│   │   ├── WatermarkEmbed.tsx
│   │   ├── WatermarkExtract.tsx
│   │   └── WatermarkConfig.tsx
│   └── App.tsx            # 主应用
└── package.json
```

## API端点

### 健康检查
- `GET /health/` - 健康检查
- `GET /health/ready` - 就绪检查
- `GET /health/live` - 存活检查

### 文件管理
- `POST /api/v1/files/upload` - 简单文件上传
- `POST /api/v1/files/upload/chunked/init` - 初始化分片上传
- `POST /api/v1/files/upload/chunked/{upload_id}/chunk/{chunk_index}` - 上传分片
- `POST /api/v1/files/upload/chunked/{upload_id}/complete` - 完成上传
- `GET /api/v1/files/upload/chunked/{upload_id}/status` - 获取上传状态
- `DELETE /api/v1/files/upload/chunked/{upload_id}` - 取消上传
- `GET /api/v1/files/{file_id}` - 下载文件
- `POST /api/v1/files/{file_id}/download-token` - 生成下载令牌
- `DELETE /api/v1/files/{file_id}` - 删除文件

### 水印任务
- `POST /api/v1/watermark/embed` - 创建水印嵌入任务
- `GET /api/v1/watermark/tasks/{task_id}` - 获取任务状态
- `POST /api/v1/watermark/tasks/{task_id}/cancel` - 取消任务
- `GET /api/v1/watermark/tasks` - 列出任务

### 水印提取
- `POST /api/v1/watermark/extract` - 提取水印

### 配置管理
- `POST /api/v1/configs` - 创建配置
- `GET /api/v1/configs/{config_id}` - 获取配置
- `GET /api/v1/configs` - 列出配置
- `DELETE /api/v1/configs/{config_id}` - 删除配置

### 审计日志
- `GET /api/v1/audit/logs` - 查询审计日志

## 使用方法

### 启动API网关

```bash
cd watermark-system/gateway/api
python main.py
```

或使用uvicorn:

```bash
uvicorn gateway.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 启动Web界面

```bash
cd watermark-frontend
npm install
npm run dev
```

### 访问服务

- API文档: http://localhost:8000/docs
- Web界面: http://localhost:3000
- Prometheus指标: http://localhost:8000/metrics
- 健康检查: http://localhost:8000/health/

## 配置

### 环境变量

创建 `.env` 文件:

```env
# API配置
API_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
DEBUG=False

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# 限流
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# JWT
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# 服务发现
BUSINESS_SERVICE_URL=http://localhost:8001
STORAGE_SERVICE_URL=http://localhost:8002
ENGINE_SERVICE_URL=http://localhost:8003

# 文件上传
MAX_UPLOAD_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=[".pdf", ".docx", ".xlsx", ".jpg", ".png", ".mp4"]
```

## 测试

### 运行API网关测试

```bash
cd watermark-system
pytest tests/gateway/test_api_gateway.py -v
pytest tests/gateway/test_file_service.py -v
```

### 运行前端测试

```bash
cd watermark-frontend
npm test
```

## 中间件

### 1. 日志中间件 (LoggingMiddleware)
- 记录所有请求和响应
- 生成唯一请求ID
- 记录请求耗时

### 2. 限流中间件 (RateLimitMiddleware)
- 基于Redis的分布式限流
- 支持按用户或IP限流
- 可配置限流阈值

### 3. 认证中间件 (AuthenticationMiddleware)
- JWT令牌验证
- 公开路径白名单
- 用户信息注入

### 4. 监控中间件 (MetricsMiddleware)
- Prometheus指标收集
- 请求计数
- 响应时间统计
- 活跃请求数

## 安全特性

1. **文件验证**
   - 扩展名检查
   - 魔数验证
   - 文件大小限制
   - 安全扫描

2. **访问控制**
   - JWT认证
   - 角色权限检查
   - 临时下载令牌
   - 一次性令牌

3. **限流保护**
   - 防止DDoS攻击
   - 按用户限流
   - 可配置阈值

4. **数据加密**
   - HTTPS传输
   - 敏感数据加密
   - 安全的密钥管理

## 性能优化

1. **分片上传**
   - 支持大文件上传
   - 断点续传
   - 并行上传

2. **缓存策略**
   - Redis缓存
   - 会话管理
   - 限流计数

3. **异步处理**
   - 异步文件操作
   - 非阻塞I/O
   - 并发请求处理

## 监控和运维

### Prometheus指标

访问 `/metrics` 端点查看:

- `api_requests_total` - 总请求数
- `api_request_duration_seconds` - 请求耗时
- `api_active_requests` - 活跃请求数
- `api_rate_limit_exceeded_total` - 限流触发次数

### 健康检查

- `/health/` - 基本健康检查
- `/health/ready` - 就绪检查（检查依赖服务）
- `/health/live` - 存活检查（简单响应）

### 日志

使用structlog进行结构化日志记录:

```python
logger.info(
    "Request completed",
    request_id=request_id,
    method=request.method,
    path=request.url.path,
    status_code=response.status_code,
    duration=f"{duration:.3f}s"
)
```

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t watermark-api-gateway -f gateway/api/Dockerfile .

# 运行容器
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  watermark-api-gateway
```

### Kubernetes部署

参考 `gateway/api/k8s/` 目录中的配置文件。

## 故障排查

### 常见问题

1. **限流触发**
   - 检查 `RATE_LIMIT_REQUESTS` 配置
   - 查看 `/metrics` 中的限流指标
   - 检查Redis连接

2. **文件上传失败**
   - 检查文件大小限制
   - 验证文件类型
   - 查看上传日志

3. **认证失败**
   - 检查JWT令牌有效性
   - 验证密钥配置
   - 查看认证日志

## 贡献

欢迎提交Issue和Pull Request。

## 许可证

[MIT License](LICENSE)
