"""
任务协议单元测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pytest
from business.task_scheduler.task_protocol import (
    WatermarkTaskMessage,
    FileCategory,
    WatermarkAction,
    TaskPriority,
    should_use_async,
    ASYNC_THRESHOLD_BYTES,
    QUEUE_ROUTING,
)


class TestWatermarkTaskMessage:
    """测试标准化任务消息"""

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
        )
        defaults.update(overrides)
        return WatermarkTaskMessage(**defaults)

    def test_serialize_deserialize(self):
        msg = self._make_msg(watermark_bits="0" * 64, visible_text="hello")
        raw = msg.to_json()
        restored = WatermarkTaskMessage.from_json(raw)
        assert restored.task_id == "t001"
        assert restored.watermark_bits == "0" * 64
        assert restored.visible_text == "hello"
        assert restored.file_category == "video"

    def test_routing_key(self):
        assert self._make_msg(file_category="video").routing_key == "watermark.video"
        assert self._make_msg(file_category="audio").routing_key == "watermark.audio"
        assert self._make_msg(file_category="document").routing_key == "watermark.document"
        assert self._make_msg(file_category="image").routing_key == "watermark.image"

    def test_default_values(self):
        msg = self._make_msg()
        assert msg.action == WatermarkAction.EMBED.value
        assert msg.priority == TaskPriority.NORMAL.value
        assert msg.max_retries == 3
        assert msg.retry_count == 0
        assert msg.timeout_seconds == 3600

    def test_json_roundtrip_with_visible_config(self):
        cfg = {"font_size": 36, "color": "#FF0000", "opacity": 0.4}
        msg = self._make_msg(visible_config=cfg)
        restored = WatermarkTaskMessage.from_json(msg.to_json())
        assert restored.visible_config == cfg


class TestShouldUseAsync:
    """测试异步分发判断"""

    def test_video_above_threshold(self):
        threshold = ASYNC_THRESHOLD_BYTES[FileCategory.VIDEO]
        assert should_use_async("video", threshold) is True
        assert should_use_async("video", threshold + 1) is True

    def test_video_below_threshold(self):
        threshold = ASYNC_THRESHOLD_BYTES[FileCategory.VIDEO]
        assert should_use_async("video", threshold - 1) is False

    def test_audio_threshold(self):
        threshold = ASYNC_THRESHOLD_BYTES[FileCategory.AUDIO]
        assert should_use_async("audio", threshold) is True
        assert should_use_async("audio", threshold - 1) is False

    def test_image_high_threshold(self):
        # 图片阈值 100MB，一般不走异步
        assert should_use_async("image", 5_000_000) is False
        assert should_use_async("image", 100 * 1024 * 1024) is True

    def test_unknown_category_defaults_async(self):
        assert should_use_async("unknown_type", 1) is True


class TestQueueRouting:
    """测试队列路由映射"""

    def test_all_categories_have_queues(self):
        for cat in FileCategory:
            assert cat in QUEUE_ROUTING
