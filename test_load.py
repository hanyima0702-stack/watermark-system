import sys
import traceback

# Try to execute the file directly
try:
    with open('engines/image/pca_watermark.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Try to compile it
    compiled = compile(code, 'engines/image/pca_watermark.py', 'exec')
    print("File compiles successfully")
    
    # Try to execute it
    namespace = {}
    exec(compiled, namespace)
    print("File executes successfully")
    print("Defined names:", [k for k in namespace.keys() if not k.startswith('__')])
    
except Exception as e:
    print("Error:")
    traceback.print_exc()
