import tensorflow as tf
import tensorflow.keras as keras




def focal_loss_binary(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Clip predictions to prevent log(0)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Compute cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

    # Compute focal weight
    prob_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = tf.pow((1.0 - prob_t), gamma)

    # Compute focal loss
    loss = alpha_factor * modulating_factor * cross_entropy
    return tf.reduce_mean(loss)





def focal_loss_categorical(y_true, y_pred, alpha=0.25, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Compute cross entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Compute focal weight
    modulating_factor = tf.pow(1.0 - y_pred, gamma)
    alpha_factor = y_true * alpha

    loss = alpha_factor * modulating_factor * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


