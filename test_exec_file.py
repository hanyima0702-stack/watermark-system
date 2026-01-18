#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import traceback

print("Attempting to execute pca_watermark.py directly...")

try:
    # Read the file
    with open('engines/image/pca_watermark.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    print(f"File read successfully, {len(code)} characters")
    print(f"First 100 chars: {code[:100]}")
    
    # Try to compile
    try:
        compiled = compile(code, 'pca_watermark.py', 'exec')
        print("File compiled successfully")
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        raise
    
    # Try to execute
    namespace = {'__name__': '__main__'}
    try:
        exec(compiled, namespace)
        print("File executed successfully")
        print(f"Defined names: {[k for k in namespace.keys() if not k.startswith('_')]}")
    except Exception as e:
        print(f"Runtime error during execution: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
