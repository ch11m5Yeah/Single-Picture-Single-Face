import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# ========== 解码预测框 ==========
def integral(distribution):
    """积分模块：将分布回归转为期望值"""
    reg_max = distribution.shape[-1] - 1
    positions = tf.range(reg_max + 1, dtype=tf.float32)
    return tf.reduce_sum(distribution * positions, axis=-1)


def decode_face_bbox(raw_out, reg_max):
    """
    解码网络输出，返回 (B, 5) -> [conf, xmin, ymin, xmax, ymax]
    """
    B = tf.shape(raw_out)[0]
    H = tf.shape(raw_out)[1]
    W = tf.shape(raw_out)[2]

    conf_pred = tf.sigmoid(raw_out[..., 0])  # (B, H, W)
    dist_pred = raw_out[..., 1:1 + 4 * (reg_max + 1)]
    dist_pred = tf.reshape(dist_pred, [B, H, W, 4, reg_max + 1])

    conf_flat = tf.reshape(conf_pred, [B, -1])
    max_idx = tf.argmax(conf_flat, axis=1, output_type=tf.int32)

    max_h = max_idx // W
    max_w = max_idx % W
    batch_idx = tf.range(B, dtype=tf.int32)

    dist_sel = tf.gather_nd(dist_pred, tf.stack([batch_idx, max_h, max_w], axis=1))
    pred_dist = integral(tf.nn.softmax(dist_sel, axis=-1))

    grid_cx = (tf.cast(max_w, tf.float32) + 0.5) / tf.cast(W, tf.float32)
    grid_cy = (tf.cast(max_h, tf.float32) + 0.5) / tf.cast(H, tf.float32)

    left_dist = pred_dist[:, 0] / reg_max
    top_dist = pred_dist[:, 1] / reg_max
    right_dist = pred_dist[:, 2] / reg_max
    bottom_dist = pred_dist[:, 3] / reg_max

    xmin = tf.clip_by_value(grid_cx - left_dist, 0.0, 1.0)
    ymin = tf.clip_by_value(grid_cy - top_dist, 0.0, 1.0)
    xmax = tf.clip_by_value(grid_cx + right_dist, 0.0, 1.0)
    ymax = tf.clip_by_value(grid_cy + bottom_dist, 0.0, 1.0)

    xmax = tf.maximum(xmax, xmin + 0.01)
    ymax = tf.maximum(ymax, ymin + 0.01)

    conf = tf.gather_nd(conf_pred, tf.stack([batch_idx, max_h, max_w], axis=1))

    return tf.stack([conf, xmin, ymin, xmax, ymax], axis=1)


# ========== IoU 计算 ==========
def bbox_iou(boxes1, boxes2):
    xmin_inter = tf.maximum(boxes1[:, 0], boxes2[:, 0])
    ymin_inter = tf.maximum(boxes1[:, 1], boxes2[:, 1])
    xmax_inter = tf.minimum(boxes1[:, 2], boxes2[:, 2])
    ymax_inter = tf.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_w = tf.maximum(xmax_inter - xmin_inter, 0.0)
    inter_h = tf.maximum(ymax_inter - ymin_inter, 0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1 + area2 - inter_area
    return inter_area / tf.maximum(union_area, 1e-10)


# ========== 自定义 Callback ==========
class IoUCallback(keras.callbacks.Callback):
    def __init__(self, val_dataset, reg_max, log_dir=None):
        super().__init__()
        self.val_dataset = val_dataset
        self.reg_max = reg_max
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir) if log_dir else None

    def on_epoch_end(self, epoch, logs=None):
        ious = []
        conf_mae = []
        for images, labels in self.val_dataset:
            preds = self.model(images, training=False)  # (B, H, W, C)
            decoded_preds = decode_face_bbox(preds, self.reg_max)  # (B, 5): [conf, xmin, ymin, xmax, ymax]

            # 获取真实框，假设 labels 形状是 (B, 1 + 4 + nc)
            gt_boxes = labels[:, 1:5]  # [xmin, ymin, xmax, ymax]
            gt_conf = labels[:, 0]
            # IoU
            batch_ious = bbox_iou(decoded_preds[:, 1:5], gt_boxes)
            batch_conf_mae = tf.abs(decoded_preds[:, 0]- gt_conf)
            ious.extend(batch_ious.numpy())
            conf_mae.extend(batch_conf_mae.numpy())
        mean_iou = sum(ious) / len(ious)
        mean_conf_iou = sum(conf_mae) / len(conf_mae)
        print(f"\nEpoch {epoch + 1}: Validation Mean IoU = {mean_iou:.4f} Validation Mean Conf MAE = {mean_conf_iou:.4f}")

        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("Validation/Mean_IoU", mean_iou, step=epoch)


def run_tflite_inference(tflite_model_path, test_dataset, reg_max, num_samples=5):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    input_scale, input_zero_point = input_details[0]['quantization']
    input_dtype = input_details[0]['dtype']

    print(f"[INFO] Input quantization: scale={input_scale}, zero_point={input_zero_point}, dtype={input_dtype}")

    # 检查量化参数是否合理
    if input_scale == 0.0:
        print("[WARNING] Input scale is 0, this might indicate no quantization or incorrect quantization info")

    for images, labels in test_dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()

        batch_size = images_np.shape[0]
        num_show = min(num_samples, batch_size)

        for i in range(num_show):
            img = images_np[i]  # shape (H,W,C), 假设范围是[-1,1]或[0,1]

            # 训练时输入范围是[-1,1]，量化到int8的[-128,127]
            if input_dtype == np.int8:
                # 线性映射：[-1,1] -> [-128,127]
                # 公式：int8_val = (float_val + 1.0) / 2.0 * 255.0 - 128.0
                img_q = (img + 1.0) / 2.0 * 255.0 - 128.0
                img_q = np.round(img_q).astype(np.int8)
            else:
                # 如果不是int8，直接使用float32
                img_q = img.astype(np.float32)

            input_data = np.expand_dims(img_q, axis=0)

            # Debug: 打印量化后的统计信息
            print(f"Sample {i + 1}: Input range [{img.min():.3f}, {img.max():.3f}] -> "
                  f"Quantized range [{img_q.min()}, {img_q.max()}], dtype={img_q.dtype}")

            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()

            raw_out = interpreter.get_tensor(output_index)[0]

            # 解码预测bbox
            decoded_pred = decode_face_bbox(tf.expand_dims(raw_out, 0), reg_max).numpy()[0]
            conf, xmin, ymin, xmax, ymax = decoded_pred
            gt_box = labels_np[i, 1:5]

            h, w = img.shape[0], img.shape[1]
            pred_px = [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
            gt_px = [int(gt_box[0] * w), int(gt_box[1] * h), int(gt_box[2] * w), int(gt_box[3] * h)]

            # 显示时将[-1,1]转换为[0,1]
            img_show = (img + 1.0) / 2.0
            img_show = np.clip(img_show, 0, 1)

            plt.figure(figsize=(8, 6))
            plt.imshow((img_show * 255).astype(np.uint8))
            plt.title(f"Sample {i + 1}: Conf {conf:.2f}")
            ax = plt.gca()

            rect_pred = plt.Rectangle((pred_px[0], pred_px[1]),
                                      pred_px[2] - pred_px[0],
                                      pred_px[3] - pred_px[1],
                                      fill=False, edgecolor='red', linewidth=2, label="Pred")
            ax.add_patch(rect_pred)

            rect_gt = plt.Rectangle((gt_px[0], gt_px[1]),
                                    gt_px[2] - gt_px[0],
                                    gt_px[3] - gt_px[1],
                                    fill=False, edgecolor='green', linewidth=2, label="GT")
            ax.add_patch(rect_gt)

            ax.legend()
            plt.show()


# 额外的调试函数：检查量化模型的输入输出详情
def debug_tflite_model(tflite_model_path):
    """调试TFLite模型的量化信息"""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("=== INPUT DETAILS ===")
    for i, detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")
        print()

    print("=== OUTPUT DETAILS ===")
    for i, detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")
        print()
def run_h5_inference(h5_model_path, test_dataset, reg_max, num_samples=5):
    # 加载 H5 模型
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    print(f"[INFO] Loaded model from {h5_model_path}")

    # 随机取一个 batch
    for images, labels in test_dataset.take(1):
        images_np = images.numpy()  # 假设 [0,1] 或 [-1,1]
        labels_np = labels.numpy()

        batch_size = images_np.shape[0]
        num_show = min(num_samples, batch_size)

        plt.figure(figsize=(15, 3 * num_show))

        for i in range(num_show):
            img = images_np[i]

            # 模型预测
            raw_out = model.predict(np.expand_dims(img, axis=0))[0]  # (H, W, C)

            # 解码预测框
            decoded_pred = decode_face_bbox(tf.expand_dims(raw_out, 0), reg_max).numpy()[0]
            conf, xmin, ymin, xmax, ymax = decoded_pred

            # 真实框
            gt_box = labels_np[i, 1:5]

            # 转像素坐标
            h, w = img.shape[0], img.shape[1]
            pred_px = [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
            gt_px = [int(gt_box[0] * w), int(gt_box[1] * h), int(gt_box[2] * w), int(gt_box[3] * h)]

            # 显示图片
            plt.subplot(num_show, 1, i + 1)
            plt.imshow((img * 255).astype(np.uint8))  # 如果 [0,1] 还原为 0-255
            plt.title(f"Sample {i+1}: Conf {conf:.2f}")
            ax = plt.gca()

            # 预测框（红色）
            rect_pred = plt.Rectangle((pred_px[0], pred_px[1]),
                                      pred_px[2] - pred_px[0],
                                      pred_px[3] - pred_px[1],
                                      fill=False, edgecolor='red', linewidth=2, label="Pred")
            ax.add_patch(rect_pred)

            # 真实框（绿色）
            rect_gt = plt.Rectangle((gt_px[0], gt_px[1]),
                                    gt_px[2] - gt_px[0],
                                    gt_px[3] - gt_px[1],
                                    fill=False, edgecolor='green', linewidth=2, label="GT")
            ax.add_patch(rect_gt)
            ax.legend()

        plt.tight_layout()
        plt.show()