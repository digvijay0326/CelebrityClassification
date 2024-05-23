import tensorflow as tf
import time
import os
from ImageClassifier.entity.config_entity import PrepareCallbackConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config
       
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    @property
    def _create_model_checkpoint_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_weights_only=True,
            save_best_only=True
        )
    @property
    def lr_rate_checkpoint_callback(self):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            verbose=1,
            min_lr=0.0001
        )
    
    def get_tb_ckpt_callback(self):
        return [
            self._create_tb_callbacks,
            self._create_model_checkpoint_callback,
            self.lr_rate_checkpoint_callback
        ]