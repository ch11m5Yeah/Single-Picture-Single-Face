import tensorflow as tf

def focal_loss_binary(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    y_true: [B, H*W]
    y_pred: [B, H*W]
    """
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
    focal_weight = alpha_factor * tf.pow(1 - pt, gamma)
    loss = -focal_weight * tf.math.log(pt)
    return loss

def distribution_focal_loss(pred, target, reg_max):
    """
    Distribution Focal Loss
    pred: [N, 4, reg_max+1] (raw logits)
    target: [N, 4]
    """
    pred = tf.nn.softmax(pred, axis=-1)  # 转概率
    target_left = tf.floor(target)
    target_right = target_left + 1

    weight_left = target_right - target
    weight_right = 1 - weight_left

    left_loss = -tf.math.log(tf.gather(pred, tf.cast(target_left, tf.int32), batch_dims=2))
    right_loss = -tf.math.log(tf.gather(pred, tf.cast(target_right, tf.int32), batch_dims=2))

    return tf.reduce_mean(weight_left * left_loss + weight_right * right_loss)

def face_detector_loss(reg_max=7, alpha_conf=0.75, gamma_conf=2.0):
    """
    Face detector loss:
    - Focal Loss for confidence map (全图)
    - DFL for bbox regression (基于标签置信度图中的位置)
    """
    def loss_fn(y_true, y_pred):
        """
        y_true: [B, 5] => [conf, xmin, ymin, xmax, ymax] (normalized)
        y_pred: [B, H, W, C], C = 1 + 4*(reg_max+1)
        """
        B = tf.shape(y_pred)[0]
        H = tf.shape(y_pred)[1]
        W = tf.shape(y_pred)[2]
        C = tf.shape(y_pred)[3]

        # === 标签解析 ===
        label_conf = y_true[:, 0]
        label_boxes = y_true[:, 1:]  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = tf.unstack(label_boxes, axis=-1)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        # === 预测 reshape ===
        pred_t = tf.reshape(y_pred, [B, H * W, C])
        pred_conf = tf.sigmoid(pred_t[..., 0])  # [B, H*W]
        pred_bbox = tf.reshape(pred_t[..., 1:1 + 4 * (reg_max + 1)], [B, H * W, 4, reg_max + 1])

        # === 构造标签置信度图 ===
        grid_x = tf.clip_by_value(tf.cast(cx * tf.cast(W, tf.float32), tf.int32), 0, W - 1)
        grid_y = tf.clip_by_value(tf.cast(cy * tf.cast(H, tf.float32), tf.int32), 0, H - 1)
        grid_idx = grid_y * W + grid_x  # [B]

        target_conf_map = tf.zeros_like(pred_conf)  # [B, H*W]
        indices = tf.stack([tf.range(B), grid_idx], axis=1)
        target_conf_map = tf.tensor_scatter_nd_update(target_conf_map, indices, label_conf)

        # === 置信度损失（全图） ===
        loss_conf = focal_loss_binary(target_conf_map, pred_conf, alpha=alpha_conf, gamma=gamma_conf)
        loss_conf = tf.reduce_mean(loss_conf)

        # === 框回归（基于标签置信度图）===
        pos_mask = target_conf_map > 0  # [B, H*W]
        pos_idx = tf.where(pos_mask)    # [[batch, hw_idx], ...]

        if tf.shape(pos_idx)[0] > 0:
            pos_pred_bbox = tf.gather_nd(pred_bbox, pos_idx)  # [N,4,reg_max+1]

            batch_ids = pos_idx[:, 0]
            gt_boxes = tf.gather(label_boxes, batch_ids)  # [N,4]
            xmin, ymin, xmax, ymax = tf.unstack(gt_boxes, axis=-1)

            hw_idx = tf.cast(pos_idx[:, 1], tf.int32)
            W_int = tf.cast(W, tf.int32)
            H_int = tf.cast(H, tf.int32)

            grid_y_idx = tf.math.floordiv(hw_idx, W_int)
            grid_x_idx = tf.math.floormod(hw_idx, W_int)

            grid_cx_norm = (tf.cast(grid_x_idx, tf.float32) + 0.5) / tf.cast(W_int, tf.float32)
            grid_cy_norm = (tf.cast(grid_y_idx, tf.float32) + 0.5) / tf.cast(H_int, tf.float32)

            l_scaled = (grid_cx_norm - xmin) * reg_max
            t_scaled = (grid_cy_norm - ymin) * reg_max
            r_scaled = (xmax - grid_cx_norm) * reg_max
            b_scaled = (ymax - grid_cy_norm) * reg_max
            target_dfl = tf.stack([l_scaled, t_scaled, r_scaled, b_scaled], axis=1)

            loss_dfl = distribution_focal_loss(pos_pred_bbox, target_dfl, reg_max)
        else:
            loss_dfl = tf.constant(0.0)
        pos_cout = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
        pos_ratio = tf.cast(pos_cout, tf.float32) / tf.cast(B*W*H, tf.float32)
        pos_ratio = tf.maximum(pos_ratio, 0.01)
        # === 总损失 ===
        return loss_conf / pos_ratio + loss_dfl * 1.5

    return loss_fn
