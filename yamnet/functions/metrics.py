import tensorflow as tf
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
import IPython

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.greater_equal(y_pred, 0.5)
        y_true = tf.greater_equal(y_true, 0.5)
        tp_values = tf.math.logical_and(y_pred, y_true)
        fp_values = tf.cast(tf.math.logical_xor(y_pred, tp_values), "float32")
        fn_values = tf.cast(tf.math.logical_xor(y_true, tp_values), "float32")
        tp_values = tf.cast(tp_values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            tp_values = tf.multiply(tp_values, sample_weight)
            fp_values = tf.multiply(fp_values, sample_weight)
            fn_values = tf.multiply(fn_values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(tp_values))
        self.false_positives.assign_add(tf.reduce_sum(fp_values))
        self.false_negatives.assign_add(tf.reduce_sum(fn_values))

    def result(self):
        # F1_Score: 2TP / (2TP + FP + FN)
        return tf.divide(tf.multiply(tf.constant(2.0), self.true_positives),
                         tf.add_n([tf.multiply(tf.constant(2.0), self.true_positives),
                                   self.false_positives, self.false_negatives]))

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()))
    
        
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)


class LWLRAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='lwlrap'):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros',
        )

        self._counts = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros',
        )
    
    def _one_sample_positive_class_precisions(self, example):
        y_true, y_pred = example

        retrieved_classes = tf.argsort(y_pred, direction='DESCENDING')
        class_rankings = tf.argsort(retrieved_classes)
        retrieved_class_true = tf.gather(y_true, retrieved_classes)
        retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))

        idx = tf.where(y_true)[:, 0]
        i = tf.boolean_mask(class_rankings, y_true)
        r = tf.gather(retrieved_cumulative_hits, i)
        c = 1 + tf.cast(i, tf.float32)
        precisions = r / c

        dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
        return dense

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=self._one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32),
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)        

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)