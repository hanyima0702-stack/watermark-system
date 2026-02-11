"""
暗水印系统使用示例

演示如何使用InvisibleWatermarkProcessor进行水印嵌入和提取。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.image.invisible_watermark import InvisibleWatermarkProcessor
from engines.image.config import WatermarkConfig


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建处理器（使用默认配置）
    processor = InvisibleWatermarkProcessor()
    
    # 准备水印数据（64位二进制字符串）
    watermark = "1010101010101010101010101010101010101010101010101010101010101010"
    
    # 嵌入水印
    print("\n嵌入水印...")
    embed_result = processor.embed_watermark(
        image_path="examples/test_image.jpg",
        watermark=watermark,
        output_path="output/watermarked.png"
    )
    
    if embed_result.success:
        print(f"✓ 嵌入成功!")
        print(f"  - 处理时间: {embed_result.processing_time:.2f}秒")
        print(f"  - 图像尺寸: {embed_result.image_size}")
        print(f"  - 宏块数量: {embed_result.block_count[0]}×{embed_result.block_count[1]}")
        print(f"  - PSNR: {embed_result.quality_metrics['psnr']:.2f}dB")
        print(f"  - SSIM: {embed_result.quality_metrics['ssim']:.4f}")
    else:
        print(f"✗ 嵌入失败: {embed_result.error_message}")
        return
    
    # 提取水印
    print("\n提取水印...")
    extract_result = processor.extract_watermark(
        image_path="output/watermarked.png"
    )
    
    if extract_result.success:
        print(f"✓ 提取成功!")
        print(f"  - 处理时间: {extract_result.processing_time:.2f}秒")
        print(f"  - 提取的水印: {extract_result.watermark_data}")
        print(f"  - 置信度: {extract_result.confidence:.3f}")
        print(f"  - 有效宏块: {extract_result.valid_blocks}/{extract_result.total_blocks}")
        print(f"  - 检测到的旋转: {extract_result.detected_rotation:.2f}°")
        print(f"  - 检测到的缩放: {extract_result.detected_scale:.2f}")
        
        # 验证水印是否匹配
        if extract_result.watermark_data == watermark:
            print("\n✓ 水印验证成功! 提取的水印与原始水印一致。")
        else:
            print("\n✗ 水印验证失败! 提取的水印与原始水印不一致。")
    else:
        print(f"✗ 提取失败: {extract_result.error_message}")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("示例2: 使用自定义配置")
    print("=" * 60)
    
    # 创建自定义配置
    config = WatermarkConfig(
        modulation_strength=12,  # 增强调制强度
        min_confidence=0.4,      # 提高最小置信度阈值
        enable_180_retry=True    # 启用180度旋转重试
    )
    
    # 创建处理器
    processor = InvisibleWatermarkProcessor(config=config)
    
    print(f"配置参数:")
    print(f"  - 调制强度: {config.modulation_strength}")
    print(f"  - 最小置信度: {config.min_confidence}")
    print(f"  - 180度重试: {config.enable_180_retry}")
    
    # 使用处理器...
    print("\n处理器已就绪，可以进行嵌入和提取操作。")


def example_config_file():
    """从配置文件加载示例"""
    print("\n" + "=" * 60)
    print("示例3: 从配置文件加载")
    print("=" * 60)
    
    # 从YAML配置文件加载
    config_path = "config/invisible_watermark_default.yaml"
    
    try:
        processor = InvisibleWatermarkProcessor(config_path=config_path)
        print(f"✓ 成功从配置文件加载: {config_path}")
        print(f"  - 宏块大小: {processor.config.block_size}")
        print(f"  - Header模式: {processor.config.header_pattern}")
        print(f"  - 调制强度: {processor.config.modulation_strength}")
    except FileNotFoundError:
        print(f"✗ 配置文件不存在: {config_path}")
    except Exception as e:
        print(f"✗ 加载配置失败: {e}")


def example_save_config():
    """保存配置到文件示例"""
    print("\n" + "=" * 60)
    print("示例4: 保存配置到文件")
    print("=" * 60)
    
    # 创建自定义配置
    config = WatermarkConfig(
        modulation_strength=15,
        min_confidence=0.35,
        scramble_seed=54321
    )
    
    # 保存到文件
    output_path = "output/my_config.yaml"
    config.to_yaml(output_path)
    print(f"✓ 配置已保存到: {output_path}")
    
    # 验证可以重新加载
    loaded_config = WatermarkConfig.from_yaml(output_path)
    print(f"✓ 配置重新加载成功")
    print(f"  - 调制强度: {loaded_config.modulation_strength}")
    print(f"  - 最小置信度: {loaded_config.min_confidence}")
    print(f"  - 加扰种子: {loaded_config.scramble_seed}")


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "=" * 60)
    print("示例5: 批量处理多个图像")
    print("=" * 60)
    
    # 创建处理器
    processor = InvisibleWatermarkProcessor()
    
    # 图像列表
    images = [
        ("examples/image1.jpg", "1010101010101010101010101010101010101010101010101010101010101010"),
        ("examples/image2.jpg", "0101010101010101010101010101010101010101010101010101010101010101"),
        ("examples/image3.jpg", "1100110011001100110011001100110011001100110011001100110011001100"),
    ]
    
    print(f"准备处理 {len(images)} 个图像...")
    
    for i, (input_path, watermark) in enumerate(images, 1):
        print(f"\n处理图像 {i}/{len(images)}: {input_path}")
        
        output_path = f"output/watermarked_{i}.png"
        
        # 嵌入水印
        result = processor.embed_watermark(input_path, watermark, output_path)
        
        if result.success:
            print(f"  ✓ 嵌入成功 (PSNR: {result.quality_metrics['psnr']:.2f}dB)")
        else:
            print(f"  ✗ 嵌入失败: {result.error_message}")


def main():
    """主函数"""
    print("\n暗水印系统使用示例")
    print("=" * 60)
    
    # 运行各个示例
    # example_basic_usage()  # 需要实际的图像文件
    example_custom_config()
    example_config_file()
    example_save_config()
    # example_batch_processing()  # 需要实际的图像文件
    
    print("\n" + "=" * 60)
    print("示例演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
