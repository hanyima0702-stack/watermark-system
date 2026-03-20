"""
任务生产者单元测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from business.task_scheduler.task_producer import TaskProducer
from business.task_scheduler.task_protocol import WatermarkTaskMessage


class TestTaskProducer:
    """测试任务生产者"""

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
        )
        defaults.update(overrides)
        return WatermarkTaskMessage(**defaults)

    @pytest.mark.asyncio
    async def test_submit_task_publishes_message(self):
        mock_mq = AsyncMock()
        producer = TaskProducer(mock_mq)

        msg = self._make_msg()
        result = await producer.submit_task(msg)

        assert result == "t001"
        mock_mq.publish.assert_called_once()
        call_kwargs = mock_mq.publish.call_args
        assert call_kwargs.kwargs["routing_key"] == "watermark.video"
        assert call_kwargs.kwargs["priority"] == msg.priority

    @pytest.mark.asyncio
    async def test_submit_task_routes_audio(self):
        mock_mq = AsyncMock()
        producer = TaskProducer(mock_mq)

        msg = self._make_msg(file_category="audio")
        await producer.submit_task(msg)

        call_kwargs = mock_mq.publish.call_args
        assert call_kwargs.kwargs["routing_key"] == "watermark.audio"

    @pytest.mark.asyncio
    async def test_submit_task_includes_correlation_id(self):
        mock_mq = AsyncMock()
        producer = TaskProducer(mock_mq)

        msg = self._make_msg()
        await producer.submit_task(msg)

        call_kwargs = mock_mq.publish.call_args
        assert call_kwargs.kwargs["correlation_id"] == msg.correlation_id
