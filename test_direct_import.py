#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("sys.path:", sys.path[:3])

try:
    print("\n1. Trying to import module...")
    import engines.image.pca_watermark as pca_module
    print("   Module imported successfully!")
    print("   Module file:", pca_module.__file__)
    print("   Module contents:", [x for x in dir(pca_module) if not x.startswith('_')])
    
    print("\n2. Trying to import PCAWatermark class...")
    from engines.image.pca_watermark import PCAWatermark
    print("   PCAWatermark imported successfully!")
    print("   PCAWatermark type:", type(PCAWatermark))
    
    print("\n3. Trying to import config...")
    from engines.image.pca_watermark import PCAWatermarkConfig
    print("   PCAWatermarkConfig imported successfully!")
    
    print("\n4. Creating instance...")
    config = PCAWatermarkConfig()
    processor = PCAWatermark(config)
    print("   Instance created successfully!")
    print("   Instance type:", type(processor))
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
