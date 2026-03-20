"""
Worker 启动入口

用法:
    # 启动全类型 Worker (监听所有队列)
    python -m business.task_scheduler.worker_main

    # 只处理视频和音频
    python -m business.task_scheduler.worker_main --queues watermark.video watermark.audio

    # 指定 RabbitMQ 地址
    python -m business.task_scheduler.worker_main --rabbitmq-url amqp://user:pass@host:5672/
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from business.task_scheduler.task_consumer import WatermarkWorker


def main():
    parser = argparse.ArgumentParser(description="Watermark Worker Node")
    parser.add_argument(
        "--queues", nargs="*", default=None,
        help="要监听的队列列表，默认监听全部",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
        help="RabbitMQ 连接 URL",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis 连接 URL",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    worker = WatermarkWorker(
        rabbitmq_url=args.rabbitmq_url,
        queues=args.queues,
        redis_url=args.redis_url,
    )

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logging.info("Worker 被用户中断")


if __name__ == "__main__":
    main()
