import tensorflow as tf
import numpy as np
import os

# ---------------- 参数配置 ----------------
INPUT_SHAPE = (1, 60, 80, 3)       # (batch, H, W, C)
REPRESENTATIVE_SAMPLES = 2000      # 校准样本总数
MODEL_H5_PATH = "best_model.h5"  # 模型路径
TFLITE_OUTPUT_PATH = "spsf5.tflite" # 输出路径
USE_REAL_DATA = True               # True=用真实验证集, False=用随机数据

# 真实数据集路径和参数（仅当 USE_REAL_DATA=True 时生效）
JSON_DIR = r"D:\Dataset\faceV2\Labels"
INPUT_SHAPE_NO_BATCH = (60, 80, 3)
BATCH_SIZE = 64

# ---------------- 1. 加载原始模型 ----------------
model = tf.keras.models.load_model(MODEL_H5_PATH, compile=False)
print(f"✅ Loaded model: Input={model.input_shape}, Output={model.output_shape}")

# ---------------- 2. 数据集加载（如果使用真实数据） ----------------
if USE_REAL_DATA:
    from dataset import generate_face_datasets
    train_dataset, valid_dataset, _ = generate_face_datasets(
        JSON_DIR,
        input_shape=INPUT_SHAPE_NO_BATCH,
        batch_size=BATCH_SIZE,
        train_ratio=0.98,
        val_ratio=0.01,
        test_ratio=0.01
    )
    print("✅ Validation dataset loaded.")

# ---------------- 3. 定义代表性数据生成器 ----------------
def representative_dataset():
    if USE_REAL_DATA:
        # 使用验证集
        total_count = 0
        max_batches = REPRESENTATIVE_SAMPLES // BATCH_SIZE
        for image_batch, _ in train_dataset.take(max_batches):
            for img in image_batch:
                img = tf.cast(img, tf.float32)
                img = tf.expand_dims(img, axis=0)  # (1,H,W,C)
                yield [img]
                total_count += 1
                if total_count >= REPRESENTATIVE_SAMPLES:
                    return
    else:
        # 使用随机数据
        for _ in range(REPRESENTATIVE_SAMPLES):
            data = np.random.uniform(low=-1.0, high=1.0, size=INPUT_SHAPE).astype(np.float32)
            yield [data]

# ---------------- 4. 配置量化转换器 ----------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 配置输入输出类型
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32  # 也可以改成 tf.int8

# ---------------- 5. 执行量化并保存 ----------------
try:
    tflite_quant_model = converter.convert()
    print("✅ Quantization successful!")
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"✅ Saved quantized model to: {os.path.abspath(TFLITE_OUTPUT_PATH)}")
except Exception as e:
    print(f"❌ Quantization failed: {e}")

# ---------------- 6. 验证量化模型 ----------------
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("\n✅ Quantization Details:")
print(f"Input: name={input_details['name']}, dtype={input_details['dtype']}, shape={input_details['shape']}")
print(f"Output: name={output_details['name']}, dtype={output_details['dtype']}, shape={output_details['shape']}")

print("\nTensor Quantization Info:")
for tensor in interpreter.get_tensor_details():
    if 'quantization' in tensor and tensor['quantization']:
        scale, zero_point = tensor['quantization']
        if scale != 0:  # 跳过未量化的张量
            print(f"{tensor['name']}: scale={scale}, zero_point={zero_point} (dtype={tensor['dtype']})")
