# 水印系统 Makefile
# 提供常用的开发和部署命令

.PHONY: help build start stop clean test lint format install-dev

# 默认目标
help:
	@echo "水印系统开发工具"
	@echo ""
	@echo "可用命令:"
	@echo "  build          构建所有Docker镜像"
	@echo "  start          启动开发环境"
	@echo "  stop           停止开发环境"
	@echo "  restart        重启开发环境"
	@echo "  clean          清理Docker资源"
	@echo "  test           运行测试"
	@echo "  lint           代码检查"
	@echo "  format         代码格式化"
	@echo "  install-dev    安装开发依赖"
	@echo "  logs           查看服务日志"
	@echo "  shell          进入开发容器"

# 构建基础镜像
build-base:
	@echo "构建基础镜像..."
	docker build -f Dockerfile.base -t watermark-system-base:latest .

# 构建所有镜像
build: build-base
	@echo "构建所有服务镜像..."
	docker-compose build

# 启动开发环境
start:
	@echo "启动开发环境..."
	docker-compose up -d
	@echo "等待服务启动..."
	sleep 10
	@echo "服务状态:"
	docker-compose ps

# 停止开发环境
stop:
	@echo "停止开发环境..."
	docker-compose down

# 重启开发环境
restart: stop start

# 清理Docker资源
clean:
	@echo "清理Docker资源..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# 查看服务日志
logs:
	docker-compose logs -f

# 查看特定服务日志
logs-api:
	docker-compose logs -f api-gateway

logs-web:
	docker-compose logs -f web-interface

logs-scheduler:
	docker-compose logs -f task-scheduler

# 进入开发容器
shell:
	docker-compose exec api-gateway /bin/bash

# 数据库操作
db-init:
	@echo "初始化数据库..."
	docker-compose exec postgres psql -U watermark_user -d watermark_system -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	@echo "运行数据库迁移..."
	docker-compose exec api-gateway alembic upgrade head

db-shell:
	docker-compose exec postgres psql -U watermark_user -d watermark_system

# 安装开发依赖
install-dev:
	@echo "安装开发依赖..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# 运行测试
test:
	@echo "运行单元测试..."
	pytest tests/ -v --cov=. --cov-report=html

test-integration:
	@echo "运行集成测试..."
	pytest tests/integration/ -v

# 代码检查
lint:
	@echo "运行代码检查..."
	flake8 .
	mypy .
	black --check .

# 代码格式化
format:
	@echo "格式化代码..."
	black .
	isort .

# 安全检查
security:
	@echo "运行安全检查..."
	bandit -r . -x tests/
	safety check

# 性能测试
benchmark:
	@echo "运行性能测试..."
	pytest tests/performance/ -v

# 生成API文档
docs:
	@echo "生成API文档..."
	docker-compose exec api-gateway python -c "from gateway.api.main import app; import json; print(json.dumps(app.openapi(), indent=2))" > docs/api.json

# 备份数据
backup:
	@echo "备份数据库..."
	docker-compose exec postgres pg_dump -U watermark_user watermark_system > backup_$(shell date +%Y%m%d_%H%M%S).sql

# 恢复数据
restore:
	@echo "恢复数据库..."
	@read -p "请输入备份文件路径: " backup_file; \
	docker-compose exec -T postgres psql -U watermark_user watermark_system < $$backup_file

# 监控服务状态
status:
	@echo "服务状态:"
	docker-compose ps
	@echo ""
	@echo "资源使用:"
	docker stats --no-stream

# 更新依赖
update-deps:
	@echo "更新Python依赖..."
	pip-compile requirements.in
	pip-compile requirements-dev.in

# 部署到生产环境
deploy-prod:
	@echo "部署到生产环境..."
	@echo "请确保已经设置了生产环境配置"
	docker-compose -f docker-compose.prod.yml up -d

# 健康检查
health:
	@echo "检查服务健康状态..."
	curl -f http://localhost:8000/health || echo "API Gateway 不健康"
	curl -f http://localhost:3000/health || echo "Web Interface 不健康"

# 生成密钥
generate-keys:
	@echo "生成系统密钥..."
	python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
	python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"

# 初始化开发环境
init-dev: install-dev build start db-migrate
	@echo "开发环境初始化完成!"
	@echo "API Gateway: http://localhost:8000"
	@echo "Web Interface: http://localhost:3000"
	@echo "MinIO Console: http://localhost:9001"
	@echo "RabbitMQ Management: http://localhost:15672"