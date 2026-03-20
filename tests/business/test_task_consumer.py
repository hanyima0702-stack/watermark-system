"""
Worker 消费者单元测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from business.task_scheduler.task_consumer import WatermarkWorker
from business.task_scheduler.task_protocol import (
    WatermarkTaskMessage, FileCategory, WatermarkAction,
)


class TestWatermarkWorker:
    """测试 Worker 消费者"""

    def _make_msg(self, **overrides):
        defaults = dict(
            task_id="t001",
            user_id="u001",
            file_id="f001",
            file_category="video",
            original_filename="demo.mp4",
            file_size=50_000_000,
            minio_bucket="watermark-videos",
            minio_object_key="u001/demo.mp4",
            watermark_bits="0" * 64,
            action=WatermarkAction.EMBED.value,
        )
        defaults.update(overrides)
        return WatermarkTaskMessage(**defaults)

    @pytest.mark.asyncio
    async def test_on_message_ack_on_success(self):
        """成功处理后应 ack 消息"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None

        # Mock _process_task 返回成功
        worker._process_task = AsyncMock(return_value={"success": True, "minio_object_key": "bucket/key"})

        msg = self._make_msg()
        incoming = AsyncMock()
        incoming.body = msg.to_json().encode("utf-8")
        incoming.headers = {"retry_count": 0}

        await worker._on_message(incoming)

        incoming.ack.assert_called_once()
        incoming.nack.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_ack_on_failure_too(self):
        """处理失败也应 ack 消息（避免无限重试循环）"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None

        worker._process_task = AsyncMock(side_effect=RuntimeError("boom"))

        msg = self._make_msg()
        incoming = AsyncMock()
        incoming.body = msg.to_json().encode("utf-8")
        incoming.headers = {"retry_count": 0}

        await worker._on_message(incoming)

        # 失败也 ack，不再 nack/reject
        incoming.ack.assert_called_once()
        incoming.nack.assert_not_called()
        incoming.reject.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_task_status_writes_redis(self):
        """状态更新应写入 Redis"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None

        await worker._update_task_status("t001", "processing", progress=50.0)

        worker._redis.set.assert_called_once()
        call_args = worker._redis.set.call_args
        key = call_args.args[0]
        assert key == "task:t001:status"
        data = json.loads(call_args.args[1])
        assert data["status"] == "processing"
        assert data["progress"] == 50.0

    def test_normalize_opacity(self):
        assert WatermarkWorker._normalize_opacity(None) == 0.4
        assert WatermarkWorker._normalize_opacity(0.5) == 0.5
        assert WatermarkWorker._normalize_opacity(50) == 0.5
        assert WatermarkWorker._normalize_opacity(100) == 1.0
