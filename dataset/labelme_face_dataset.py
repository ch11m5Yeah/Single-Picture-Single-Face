import os
import json
import tensorflow as tf
import numpy as np
from labelme.utils import img_b64_to_arr
import random

# 类别映射
LABEL_KEEP = ["face", "half_face"]
LABEL_NOFACE = "noface"

def parse_labelme_json(json_path):
    """
    解析 LabelMe JSON 文件，提取图像和归一化 bbox。
    y_true 格式: [conf, xmin, ymin, xmax, ymax]
    - 如果存在 'noface' 标签，则返回全零标签。
    - 如果有 face 或 half_face，取第一个目标。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 解码图像
        if not data.get('imageData'):
            raise ValueError("缺少 imageData")
        img = img_b64_to_arr(data['imageData'])
        if img is None:
            raise ValueError("图像解码失败")
        img_height, img_width = img.shape[:2]

        # 初始化 label
        label = np.zeros((1 + 4,), dtype=np.float32)  # [conf, xmin, ymin, xmax, ymax]

        shapes = data.get('shapes', [])
        if len(shapes) == 0:
            return img, label  # 没有目标

        # 如果出现 noface，直接返回全零
        for shape in shapes:
            if shape['label'] == LABEL_NOFACE:
                return img, label  # 全部置零

        # 查找第一个 face 或 half_face
        for shape in shapes:
            if shape['label'] in LABEL_KEEP:
                points = shape.get('points', [])
                if len(points) < 2:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                # 归一化坐标
                label[0] = 1.0
                label[1] = x_min / img_width
                label[2] = y_min / img_height
                label[3] = x_max / img_width
                label[4] = y_max / img_height

                break  # 只取第一个有效目标

        return img, label

    except Exception as e:
        print(f"解析 {json_path} 失败: {e}")
        return None, None


def load_data_from_json(json_files):
    """
    从 JSON 文件加载所有图像和标签。
    返回:
        imgs: numpy array，shape (N, H, W, 3)
        labels: numpy array，shape (N, 5+NC)
    """
    imgs, labels = [], []
    for file in json_files:
        img, label = parse_labelme_json(file)
        if img is None:
            continue
        imgs.append(img)
        labels.append(label)
    return np.array(imgs, dtype=np.uint8), np.array(labels, dtype=np.float32)


def create_tf_face_dataset(json_files, img_size=(224, 224), batch_size=32, shuffle=True):
    """
    使用 from_tensor_slices 创建数据集。
    """
    imgs, labels = load_data_from_json(json_files)
    print(f"加载完成: {len(imgs)} 张图片")

    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))

    # 预处理
    def preprocess(img, target):
        img = tf.image.resize(img, [img_size[0], img_size[1]])  # [H, W]
        img = (tf.cast(img, tf.float32) / 255.0 - 0.5) * 2  # [-1, 1]
        return img, target

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(imgs))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def generate_face_datasets(json_dir, input_shape, batch_size,
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"目录 {json_dir} 中没有 JSON 文件")

    random.shuffle(json_files)
    total = len(json_files)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_files = json_files[:train_size]
    val_files = json_files[train_size:train_size + val_size]
    test_files = json_files[train_size + val_size:]

    print(f"训练集: {len(train_files)} 文件, 验证集: {len(val_files)} 文件, 测试集: {len(test_files)} 文件")

    train_dataset = create_tf_face_dataset(train_files, img_size=input_shape, batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_face_dataset(val_files, img_size=input_shape, batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_face_dataset(test_files, img_size=input_shape, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset
