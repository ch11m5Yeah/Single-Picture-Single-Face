import tensorflow as tf


def create_lr_schedule(initial_lr, eta_min, warmup_epochs, steps_per_epoch, first_decay_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch

    cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=steps_per_epoch * first_decay_epoch,
        t_mul=2.0,
        m_mul=0.6,
        alpha=eta_min / initial_lr
    )

    class WarmupThenCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self):
            super().__init__()
            self.initial_lr = initial_lr
            self.eta_min = eta_min
            self.warmup_epochs = warmup_epochs
            self.steps_per_epoch = steps_per_epoch
            self.warmup_steps = warmup_steps
            self.cosine_decay = cosine_decay

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)
            return tf.cond(
                step < warmup_steps_f,
                lambda: self.initial_lr * (step / warmup_steps_f),
                lambda: self.cosine_decay(step - warmup_steps_f)
            )

        def get_config(self):
            return {
                "initial_lr": self.initial_lr,
                "eta_min": self.eta_min,
                "warmup_epochs": self.warmup_epochs,
                "steps_per_epoch": self.steps_per_epoch,
            }

    return WarmupThenCosine()
