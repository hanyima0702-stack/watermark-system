from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from engines.media.video_invisible_watermark import VideoWatermarker
from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker
from pydub import AudioSegment
import os

class Test:

    watermark = "1010101010101010101010101010101010101010101010101010101010101010"


    def testrotate(self):


        img = Image.open("D:/pic/watermarked.png")

        # 旋转图片
        # fillcolor可以设置填充色，如'white'或(255,255,255)
        rotated_img = img.rotate(angle=166)
        rotated_img.save("D:/pic/watermarked1.png")


    def testBrightness(self):
        img = Image.open("D:/pic/watermarked.png")

        # 创建亮度增强器
        enhancer = ImageEnhance.Brightness(img)

        # 调整亮度
        adjusted = enhancer.enhance(0.01)

        # 保存图片
        adjusted.save("D:/pic/watermarked1.png")


    def testCompress(self):
        """
            使用 OpenCV 模拟 JPEG 压缩
            :param quality: 0-100，数值越低压缩越严重。
                            95: 轻微压缩 (微信原图)
                            75: 标准压缩 (普通网页)
                            50: 明显压缩 (朋友圈/缩略图)
                            10-30: 暴力压缩
            """
        quality=5

        img = cv2.imread("D:/pic/watermarked.png" )

        # cv2.IMWRITE_JPEG_QUALITY 控制压缩质量
        cv2.imwrite("D:/pic/watermarked1.jpg" , img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


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
            image_path="D:/pic/watermarked1.png"  # 带水印的图像
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



###############################################
################测试视频水印#####################

    def test_video_embed(self):
        input = "D:/video/video1.mp4"
        output = "D:/video/watermarked.mp4"

        watermarker = VideoWatermarker()
        watermarker.embed_video_timerange(input, output,self.watermark,0)



    def test_video_extract(self):
        input = "D:/video/watermarked.mp4"
        watermarker = VideoWatermarker()
        watermarker.extract_from_timerange(input,0,1)



#####################################################################
##########################################################audio


    def test_audio_embed(self):
        input = "D:/video/audio.mp3"
        output = "D:/video/watermarked.mp3"
        watermarker = AudioInvisibleWatermarker()
        watermarker.embed_file(input,output,self.watermark)

    def test_audio_extract(self):
        input = "D:/video/watermarked.mp3"
        watermarker = AudioInvisibleWatermarker()
        print(watermarker.extract_file(input,64))



    def test_cut_audio(self):
        input = "D:/video/watermarked.wav"
        output_file="D:/video/watermarked1.wav"
        audio = AudioSegment.from_file(input)
        start_time=0
        end_time=100000

        # 截取音频片段
        cut_audio = audio[start_time:end_time]

        # 导出截取后的音频
        cut_audio.export(output_file, format=output_file.split('.')[-1])

        print(f"截取成功！文件已保存至: {output_file}")
        print(f"原始时长: {len(audio) / 1000:.2f}秒")
        print(f"截取时长: {len(cut_audio) / 1000:.2f}秒")