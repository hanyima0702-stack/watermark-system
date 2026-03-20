from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from engines.media.video_invisible_watermark import VideoWatermarkProcessor
from engines.media.audio_invisible_watermark import AudioInvisibleWatermarker
from pydub import AudioSegment
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import subprocess

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

    import subprocess

    def test_cut_video_lossless(self):
        input_path = "D:/video/1111.mp4"
        start_time = 0
        end_time = 10
        output_path = "D:/video/22_lossless.mp4"

        # 构建 FFmpeg 命令进行无损流拷贝
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-i", input_path,
            "-c:v", "copy",  # 核心关键点：复制视频流，绝对不重新编码！
            "-c:a", "copy",  # 复制音频流
            output_path
        ]

        try:
            print(f"开始无损截取视频: {input_path} ...")
            # 运行命令
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"视频无损截取成功！保存至: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"截取失败！FFmpeg 报错信息:\n{e.stderr}")
            return False

    # 运行测试
    # test_cut_video_lossless()



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
        input = "D:/video/audio3.wav"
        output_file="D:/video/watermarked1.wav"
        audio = AudioSegment.from_file(input)
        start_time=0
        end_time=5000

        # 截取音频片段
        cut_audio = audio[start_time:end_time]

        # 导出截取后的音频
        cut_audio.export(output_file, format=output_file.split('.')[-1])

        print(f"截取成功！文件已保存至: {output_file}")
        print(f"原始时长: {len(audio) / 1000:.2f}秒")
        print(f"截取时长: {len(cut_audio) / 1000:.2f}秒")