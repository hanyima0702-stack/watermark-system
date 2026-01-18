import traceback

try:
    import engines.image.pca_watermark as pca
    print("Module loaded successfully")
    print("Module contents:", dir(pca))
except Exception as e:
    print("Error loading module:")
    traceback.print_exc()
