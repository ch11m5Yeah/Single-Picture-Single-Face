import tensorflow as tf
import tensorflow.keras as keras
from train_opti import create_lr_schedule
from model import FaceDetector
from dataset import generate_face_datasets
from loss import face_detector_loss
from valid import *

def export_tflite_model(saved_model_dir, tflite_path, representative_data_gen=None, int8=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32
    else:
        # Float16 量化（精度高，速度快）
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] TFLite model saved to {tflite_path}")

def main():
    input_shape = (60, 80, 3)
    json_dir = r"D:\Dataset\face_strengthen\Labels"
    reg_max = 7
    initial_lr = 1e-3
    eta_min = 1e-8
    warmup_epochs = 0
    first_decay_epoch = 150
    total_epochs = 500
    batch_size = 64
    patience = 40
    # 初始化模型
    model = FaceDetector(input_shape=input_shape, reg_max=reg_max)

    # 损失函数
    loss_fn = face_detector_loss(reg_max=reg_max)

    # 数据集
    train_dataset, val_dataset, test_dataset = generate_face_datasets(
        json_dir, input_shape=input_shape, batch_size=batch_size,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # 学习率调度
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    lr_schedule = create_lr_schedule(initial_lr=initial_lr, eta_min=eta_min,
                                     warmup_epochs=warmup_epochs,
                                     steps_per_epoch=steps_per_epoch,
                                     first_decay_epoch=first_decay_epoch)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
        keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
        keras.callbacks.TensorBoard(log_dir="./logs"),
        IoUCallback(val_dataset, reg_max, log_dir="./logs")
    ]

    # 编译模型
    model.compile(optimizer=optimizer, loss=loss_fn)

    # 训练
    model.fit(train_dataset, validation_data=val_dataset,
              epochs=total_epochs, callbacks=callbacks)

    # 加载最佳模型
    best_model = keras.models.load_model("best_model.h5", compile=False)

    # 保存 SavedModel 格式
    saved_model_dir = "face_detector_final_best"
    best_model.save(saved_model_dir)

    # ========== 使用真实数据集构建 representative_data ==========
    def representative_data_gen():
        for _ in range(500):  # 生成500个样本
            # 随机生成输入，范围[0,1]，符合模型训练时的输入分布
            img = tf.random.uniform(shape=(1, 60, 80, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
            yield [img.numpy()]

    # ========== 导出 int8 量化模型 ==========
    export_tflite_model(saved_model_dir, "face_detector_best_int8.tflite",
                        representative_data_gen=representative_data_gen, int8=True)
    run_tflite_inference(
        tflite_model_path="face_detector_best_int8.tflite",
        test_dataset=test_dataset,
        reg_max=7,
        num_samples=20
    )
    # ========== 导出 float16 量化模型（可选） ==========
    # export_tflite_model(saved_model_dir, "face_detector_best_fp16.tflite",
    #                     representative_data_gen=None, int8=False)

if __name__ == "__main__":
    main()
