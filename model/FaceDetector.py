import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

# 空间注意力模块
def spatial_attention(inputs):
    # 生成 1 通道注意力图
    attention = layers.Conv2D(1, kernel_size=5 , strides=1, padding='same', activation='sigmoid')(inputs)
    return inputs * attention

# 带注意力的 Depthwise Separable 卷积
def depthwise_separable_conv(inputs, dw_kernel=3, pointwise_filters=64, strides=1, padding='same', use_spatial_att=False):
    # 1. Depthwise 卷积
    x = layers.DepthwiseConv2D(kernel_size=dw_kernel, strides=strides, padding=padding, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 2. Pointwise 卷积
    x = layers.Conv2D(filters=pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3. 可选空间注意力
    if use_spatial_att:
        x = spatial_attention(x)

    return x
def DownSampleConv(inputs, kernel_size, filters, stride, padding='same'):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x
# 主模型
def bottleneck_block(x, out_channels, stride=1, expansion=4):
    """
    简化版 Bottleneck Block (无残差)

    Args:
        x: 输入 Tensor
        out_channels: 最终输出通道数
        stride: 步幅 (1=保持大小, 2=下采样)
        expansion: 通道扩张倍数 (默认4)
        name: block 名称

    Returns:
        Tensor
    """
    mid_channels = out_channels // expansion

    # 1x1 Conv 降维
    x = layers.Conv2D(mid_channels, (1, 1), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 3x3 Conv
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding="same", use_bias=False)(x)
    # x = layers.Conv2D(mid_channels, (3, 3), strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 1x1 Conv 升维
    x = layers.Conv2D(out_channels, (1, 1), strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def FaceDetector(input_shape=(60, 80, 3), reg_max=7):
    inputs = keras.layers.Input(shape=input_shape)
    out_channels = 1 + 4 * (reg_max + 1)

    # 添加注意力到每一层
    x = depthwise_separable_conv(inputs, 3, 16, 2, use_spatial_att=False)
    x = bottleneck_block(x, 16)
    x = depthwise_separable_conv(x, 3, 32, 2, use_spatial_att=False)
    x = bottleneck_block(x, 32)
    x = depthwise_separable_conv(x, 3, 64, 2, use_spatial_att=False)
    x = bottleneck_block(x, 64)
    x = depthwise_separable_conv(x, 3, 128, 2, use_spatial_att=False)

    # 输出卷积层
    x = layers.Conv2D(out_channels, (1, 1), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    model = keras.Model(inputs=inputs, outputs=x, name="FaceDetector")
    return model

if __name__ == "__main__":
    model = FaceDetector()
    model.summary()

    # 随机输入测试
    dummy_input = tf.random.normal((1, 60, 80, 3))
    out = model(dummy_input)
    print("\nOutput shape:", out.shape)
