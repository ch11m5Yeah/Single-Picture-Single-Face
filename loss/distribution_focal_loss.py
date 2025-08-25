import tensorflow as tf


def distribution_focal_loss(pred, target, reg_max=16):
    """
    Distribution Focal Loss in TensorFlow.

    Args:
        pred: (B, N, reg_max+1) raw logits
        target: (B, N) continuous target values
        reg_max: int
    Returns:
        scalar loss
    """
    # Apply softmax to get probabilities
    pred_prob = tf.nn.softmax(pred, axis=-1)  # (B, N, reg_max+1)

    # Floor and ceil for target
    target_left = tf.floor(target)
    target_right = target_left + 1.0

    # Weights
    weight_right = target - target_left
    weight_left = 1.0 - weight_right

    # Clamp indices
    target_left_idx = tf.clip_by_value(tf.cast(target_left, tf.int32), 0, reg_max)
    target_right_idx = tf.clip_by_value(tf.cast(target_right, tf.int32), 0, reg_max)

    # Gather probabilities
    batch_idx = tf.range(tf.shape(pred)[0])[:, None]  # (B,1)
    pos_idx = tf.range(tf.shape(pred)[1])[None, :]  # (1,N)
    batch_idx = tf.tile(batch_idx, [1, tf.shape(pred)[1]])  # (B,N)
    pos_idx = tf.tile(pos_idx, [tf.shape(pred)[0], 1])  # (B,N)

    # Flatten indices for gather_nd
    idx_left = tf.stack([batch_idx, pos_idx, target_left_idx], axis=-1)  # (B,N,3)
    idx_right = tf.stack([batch_idx, pos_idx, target_right_idx], axis=-1)

    prob_left = tf.gather_nd(pred_prob, idx_left)  # (B,N)
    prob_right = tf.gather_nd(pred_prob, idx_right)  # (B,N)

    # Compute log loss
    loss_left = -tf.math.log(tf.clip_by_value(prob_left, 1e-9, 1.0))
    loss_right = -tf.math.log(tf.clip_by_value(prob_right, 1e-9, 1.0))

    # Weighted sum
    loss = weight_left * loss_left + weight_right * loss_right
    return tf.reduce_mean(loss)