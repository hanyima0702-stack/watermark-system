# Task Scheduler Dockerfile
FROM watermark-system-base:latest

# 复制任务调度器代码
COPY business/task_scheduler/ ./business/task_scheduler/
COPY business/config_manager/ ./business/config_manager/
COPY business/audit_service/ ./business/audit_service/
COPY storage/ ./storage/

# 启动Celery Worker
CMD ["celery", "-A", "business.task_scheduler.celery_app", "worker", "--loglevel=info", "--concurrency=4"]