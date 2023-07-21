""" Copyright 2023 IBM Research. All Rights Reserved.
"""
import tensorflow as tf


class SparseMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int, name: str = 'sparse_mean_iou', **kwargs):
        super(SparseMeanIoU, self).__init__(name=name, **kwargs)
        self.tf_mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tf_mean_iou.update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)

    def result(self):
        return self.tf_mean_iou.result()

    def reset_states(self):
        self.tf_mean_iou.reset_states()
