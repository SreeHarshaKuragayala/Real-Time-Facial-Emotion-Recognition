import tensorflow as tf
import numpy as np

print("=" * 50)
print("🔧 System Information")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

print("\n🔍 GPU Detection")
print("=" * 50)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected:")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")

    # Test GPU memory
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU memory growth enabled")
    except Exception as e:
        print(f"⚠️ GPU memory setup warning: {e}")
else:
    print("❌ No GPU detected")

# Test basic operations
print("\n🧮 Quick Performance Test")
print("=" * 50)
try:
    # Create test tensors
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)

    device = "GPU" if gpus else "CPU"
    print(f"✅ Matrix multiplication test passed on {device}")
    print(f"   Result shape: {c.shape}")

except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n🎯 Ready for emotion recognition training!")