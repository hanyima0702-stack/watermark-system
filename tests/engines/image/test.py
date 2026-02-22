from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from PIL import Image
import math

class Test:

    watermark = "1010101010101010101010101010101010101010101010101010101010101010"


    def testrotate(self):


        img = Image.open("D:/pic/watermarked.png")

        # 旋转图片
        # fillcolor可以设置填充色，如'white'或(255,255,255)
        rotated_img = img.rotate(angle=166)
        rotated_img.save("D:/pic/watermarked1.png")

    def testEmbed(self):
        # 初始化处理器
        processor = InvisibleWatermarkProcessor()

        # ========== 嵌入水印 ==========

        embed_result = processor.embed_watermark(
            image_path="D:/pic/pic.jpeg",  # 原始图像
            watermark=self.watermark,  # 64位水印
            output_path="D:/pic/watermarked.png"  # 输出图像
        )

        if embed_result.success:
            print(f"✓ 嵌入成功!")
        else:
            print(f"✗ 嵌入失败: {embed_result.error_message}")


    def testExtract(self):

        processor = InvisibleWatermarkProcessor()

        # ========== 提取水印 ==========
        extract_result = processor.extract_watermark(
            image_path="D:/pic/watermarked2.jpg"  # 带水印的图像
        )

        if extract_result.success:
            print(f"✓ 提取成功!")
            print(f"  水印: {extract_result.watermark_data}")
            print(f"  置信度: {extract_result.confidence:.3f}")

            # 验证水印
            if extract_result.watermark_data == self.watermark:
                print("✓ 水印验证通过!")
            else:
                print("✗ 水印不匹配")
        else:
            print(f"✗ 提取失败: {extract_result.error_message}")

