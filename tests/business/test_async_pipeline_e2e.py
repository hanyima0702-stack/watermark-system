"""
异步处理管线端到端集成测试

验证完整链路：
  1. 任务协议序列化 → 2. 生产者投递 → 3. 消费者接收 → 4. 状态更新 → 5. 结果回写

不依赖真实 RabbitMQ / Redis / MinIO，全部 Mock。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business.task_scheduler.task_protocol import (
    WatermarkTaskMessage,
    FileCategory,
    WatermarkAction,
    TaskPriority,
    should_use_async,
    ASYNC_THRESHOLD_BYTES,
)
from business.task_scheduler.task_producer import TaskProducer
from business.task_scheduler.task_consumer import WatermarkWorker


def _make_video_task(**overrides):
    """构造一个视频嵌入任务消息"""
    defaults = dict(
        task_id="e2e_test_001",
        user_id="user_abc",
        file_id="file_xyz",
        file_category=FileCategory.VIDEO.value,
        original_filename="big_video.mp4",
        file_size=50 * 1024 * 1024,  # 50 MB
        minio_bucket="watermark-videos",
        minio_object_key="user_abc/20260319/big_video.mp4",
        action=WatermarkAction.EMBED.value,
        watermark_type="both",
        watermark_bits="0" * 64,
        visible_text="测试水印",
        output_bucket="watermark-results",
        output_object_key="results/user_abc/e2e_test_001.mp4",
        priority=TaskPriority.HIGH.value,
    )
    defaults.update(overrides)
    return WatermarkTaskMessage(**defaults)


class TestAsyncThresholdDecision:
    """测试异步分发决策逻辑"""

    def test_video_10mb_triggers_async(self):
        """10MB 视频应走异步"""
        assert should_use_async("video", 10 * 1024 * 1024) is True

    def test_video_5mb_stays_sync(self):
        """5MB 视频应走同步"""
        assert should_use_async("video", 5 * 1024 * 1024) is False

    def test_audio_20mb_triggers_async(self):
        """20MB 音频应走异步"""
        assert should_use_async("audio", 20 * 1024 * 1024) is True

    def test_audio_10mb_stays_sync(self):
        """10MB 音频应走同步"""
        assert should_use_async("audio", 10 * 1024 * 1024) is False

    def test_document_50mb_triggers_async(self):
        """50MB 文档应走异步"""
        assert should_use_async("document", 50 * 1024 * 1024) is True

    def test_image_small_stays_sync(self):
        """小图片应走同步"""
        assert should_use_async("image", 5 * 1024 * 1024) is False


class TestProducerConsumerPipeline:
    """测试生产者 → 消费者完整管线"""

    @pytest.mark.asyncio
    async def test_full_pipeline_message_integrity(self):
        """
        验证消息从生产者到消费者的完整性：
        生产者发出的 JSON 能被消费者正确解析
        """
        msg = _make_video_task()

        # 模拟生产者发布
        captured_body = None
        captured_routing_key = None

        async def mock_publish(routing_key, body, priority, correlation_id, headers):
            nonlocal captured_body, captured_routing_key
            captured_body = body
            captured_routing_key = routing_key

        mock_mq = AsyncMock()
        mock_mq.publish = mock_publish
        producer = TaskProducer(mock_mq)
        await producer.submit_task(msg)

        # 验证路由
        assert captured_routing_key == "watermark.video"

        # 模拟消费者解析
        restored = WatermarkTaskMessage.from_json(captured_body)
        assert restored.task_id == "e2e_test_001"
        assert restored.user_id == "user_abc"
        assert restored.file_category == "video"
        assert restored.watermark_bits == "0" * 64
        assert restored.visible_text == "测试水印"
        assert restored.output_bucket == "watermark-results"
        assert restored.priority == TaskPriority.HIGH.value

    @pytest.mark.asyncio
    async def test_worker_updates_redis_on_processing(self):
        """Worker 处理时应通过 Redis 上报进度"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None

        await worker._update_task_status("t001", "processing", progress=50.0)

        worker._redis.set.assert_called_once()
        key, value = worker._redis.set.call_args.args[:2]
        assert key == "task:t001:status"
        data = json.loads(value)
        assert data["status"] == "processing"
        assert data["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_worker_updates_redis_on_completion(self):
        """Worker 完成时应写入结果"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None

        result = {"success": True, "minio_object_key": "watermark-results/results/user_abc/t001.mp4"}
        await worker._update_task_status(
            "t001", "completed", progress=100.0,
            result=result, processing_time=12.5,
        )

        key, value = worker._redis.set.call_args.args[:2]
        data = json.loads(value)
        assert data["status"] == "completed"
        assert data["progress"] == 100.0
        assert data["result"]["success"] is True
        assert data["processing_time"] == 12.5

    @pytest.mark.asyncio
    async def test_worker_ack_on_success(self):
        """处理成功后消息应被 ack"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None
        worker._process_task = AsyncMock(return_value={"success": True})

        msg = _make_video_task()
        incoming = AsyncMock()
        incoming.body = msg.to_json().encode("utf-8")
        incoming.headers = {"retry_count": 0}

        await worker._on_message(incoming)

        incoming.ack.assert_called_once()
        incoming.nack.assert_not_called()
        incoming.reject.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_ack_on_failure_too(self):
        """失败也应 ack，避免无限重试循环"""
        worker = WatermarkWorker(rabbitmq_url="amqp://localhost")
        worker._redis = AsyncMock()
        worker._db_manager = None
        worker._process_task = AsyncMock(side_effect=IOError("MinIO 连接超时"))

        msg = _make_video_task()
        incoming = AsyncMock()
        incoming.body = msg.to_json().encode("utf-8")
        incoming.headers = {"retry_count": 1}

        await worker._on_message(incoming)

        # 失败也 ack，不再 nack
        incoming.ack.assert_called_once()
        incoming.nack.assert_not_called()
        incoming.reject.assert_not_called()


class TestQueueRouting:
    """测试不同文件类型路由到正确队列"""

    def test_video_routes_to_video_queue(self):
        msg = _make_video_task(file_category="video")
        assert msg.routing_key == "watermark.video"

    def test_audio_routes_to_audio_queue(self):
        msg = _make_video_task(file_category="audio")
        assert msg.routing_key == "watermark.audio"

    def test_document_routes_to_document_queue(self):
        msg = _make_video_task(file_category="document")
        assert msg.routing_key == "watermark.document"

    def test_image_routes_to_image_queue(self):
        msg = _make_video_task(file_category="image")
        assert msg.routing_key == "watermark.image"


class TestMessagePriorityAndRetry:
    """测试消息优先级和重试机制"""

    def test_priority_values(self):
        assert TaskPriority.LOW.value == 0
        assert TaskPriority.NORMAL.value == 5
        assert TaskPriority.HIGH.value == 8
        assert TaskPriority.URGENT.value == 10

    def test_high_priority_message(self):
        msg = _make_video_task(priority=TaskPriority.URGENT.value)
        assert msg.priority == 10

    def test_default_retry_config(self):
        msg = _make_video_task()
        assert msg.max_retries == 3
        assert msg.retry_count == 0
        assert msg.timeout_seconds == 3600

    @pytest.mark.asyncio
    async def test_producer_sends_priority(self):
        """生产者应将优先级传递给 RabbitMQ"""
        mock_mq = AsyncMock()
        producer = TaskProducer(mock_mq)

        msg = _make_video_task(priority=TaskPriority.URGENT.value)
        await producer.submit_task(msg)

        call_kwargs = mock_mq.publish.call_args
        assert call_kwargs.kwargs["priority"] == 10
