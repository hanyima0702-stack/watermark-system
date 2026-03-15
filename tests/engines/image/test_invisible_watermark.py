"""
暗水印主控制器测试

测试InvisibleWatermarkProcessor的基本功能。
"""

import pytest
import numpy as np
import cv2
import os
from unittest.mock import AsyncMock, MagicMock

from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from engines.image.config import WatermarkConfig


def make_mock_minio():
    """创建 MinIO mock，upload_file 为协程"""
    mock = MagicMock()
    mock.config = MagicMock()
    mock.config.result_bucket = "watermark-results"
    mock.upload_file = AsyncMock(return_value={"etag": "test-etag"})
    return mock


class TestInvisibleWatermarkProcessor:
    """测试暗水印处理器"""

    @pytest.fixture
    def processor(self):
        config = WatermarkConfig()
        return InvisibleWatermarkProcessor(config=config)

    @pytest.fixture
    def test_image(self):
        return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    @pytest.fixture
    def watermark_data(self):
        return "1010101010101010101010101010101010101010101010101010101010101010"

    def test_processor_initialization(self, processor):
        assert processor is not None
        assert processor.config is not None
        assert processor.ecc_encoder is not None
        assert processor.scrambler is not None
        assert processor.image_embedder is not None

    @pytest.mark.asyncio
    async def test_embed_watermark(self, processor, test_image, watermark_data, tmp_path):
        """测试水印嵌入并上传到 MinIO"""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), test_image)

        mock_minio = make_mock_minio()

        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path=str(input_path),
            minio_service=mock_minio,
        )

        assert result.success is True
        assert result.watermark_data == watermark_data
        assert result.minio_object_key is not None
        assert result.processing_time > 0
        mock_minio.upload_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_with_invalid_image(self, processor, watermark_data):
        """测试无效图像的嵌入"""
        mock_minio = make_mock_minio()

        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path="/nonexistent/path/image.png",
            minio_service=mock_minio,
        )

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_embed_with_small_image(self, processor, watermark_data, tmp_path):
        """测试小图像的嵌入"""
        small_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        input_path = tmp_path / "small.png"
        cv2.imwrite(str(input_path), small_image)

        mock_minio = make_mock_minio()

        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path=str(input_path),
            minio_service=mock_minio,
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_embed_without_minio_raises(self, processor, test_image, watermark_data, tmp_path):
        """不传 minio_service 应该失败"""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), test_image)

        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path=str(input_path),
        )

        assert result.success is False
        assert "minio_service" in result.error_message

    @pytest.mark.asyncio
    async def test_embed_custom_object_key(self, processor, test_image, watermark_data, tmp_path):
        """测试自定义 object_key"""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), test_image)

        mock_minio = make_mock_minio()
        custom_key = "custom/path/result.png"

        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path=str(input_path),
            minio_service=mock_minio,
            object_key=custom_key,
        )

        assert result.success is True
        assert custom_key in result.minio_object_key

    def test_extract_with_invalid_image(self, processor, tmp_path):
        """测试无效图像的提取"""
        result = processor.extract_watermark(str(tmp_path / "nonexistent.png"))

        assert result.success is False
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_embed_result_structure(self, processor, test_image, watermark_data, tmp_path):
        """测试嵌入结果包含所有必要字段"""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), test_image)

        mock_minio = make_mock_minio()
        result = await processor.embed_watermark(
            watermark=watermark_data,
            image_path=str(input_path),
            minio_service=mock_minio,
        )

        assert hasattr(result, 'success')
        assert hasattr(result, 'watermark_data')
        assert hasattr(result, 'encoded_data')
        assert hasattr(result, 'block_count')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'image_size')
        assert hasattr(result, 'minio_object_key')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
