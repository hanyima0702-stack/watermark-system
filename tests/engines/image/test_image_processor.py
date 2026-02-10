import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from engines.document.pdf_processor import PDFProcessor
from engines.document.base_processor import WatermarkConfig, DocumentProcessingError, WatermarkEmbeddingError
from engines.image.image_processor import ImageProcessor
from engines.image.recover import estimate_crop_parameters, recover_crop


class TestImageProcessor:
    @pytest.fixture
    def processor(self):
        """Create PDFProcessor instance."""
        return ImageProcessor()

    @pytest.fixture
    def watermark(self):
        return "D:/pic/watermark_pic.png"


    def test_add_watermark(selfs,processor,watermark):
        processor.read_img("D:/pic/ori_img.jpeg")

        processor.read_wm(watermark)

        processor.embed("D:/pic/output.png")

    def test_extract(self,processor):
        (x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(
            original_file='D:/pic/input1.png',
            template_file='D:/pic/input4.png',
            scale=(0.5, 2), search_num=200)

        recover_crop(template_file='D:/pic/input4.png', output_file_name='D:/pic/input4_还原.png',
                     loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)

        processor.extract(filename="D:/pic/input4.png",wm_shape=(64,64),out_wm_name="D:/pic/extracted4.png")