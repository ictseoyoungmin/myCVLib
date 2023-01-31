import tensorflow as tf
import numpy as np

# https://ndb796.tistory.com/653 # for InstanceNormalize
def calc_mean_std(feat, eps=1e-5):
    n, c = feat.shape[0], 1
    feat_var = np.var(feat.reshape(n, c, -1), axis=2) + eps
    feat_std = tf.sqrt(feat_var).reshape(n, c, 1, 1)
    feat_mean = tf.math.reduce_mean(feat.reshape(n, c, -1), axis=2).reshape(n, c, 1, 1)
    
    return feat_mean, feat_std

# custom normalize layer : tensorflow에 inm 없음
class InstanceNormalize(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNormalize, self).__init__()
        if self._compute_dtype not in ("float16", "bfloat16", "float32", None):
            raise ValueError(
                "Passing `fused=True` is only supported when the compute "
                "dtype is float16, bfloat16, or float32. Got dtype: %s"
                % (self._compute_dtype,)
            )
    def instanceNormalize(self,feat,shape, eps=1e-5): # c : local var
        n = shape[0]
        c = 1
        feat_var = tf.math.reduce_variance(tf.reshape(feat,(n, c, -1)), axis=2) + eps
        feat_std = tf.reshape(tf.sqrt(feat_var),(n, c, 1))
        feat_mean = tf.reshape(tf.math.reduce_mean(tf.reshape(feat,(n, c, -1)), axis=2),(n, c, 1))

        out = (feat - feat_mean) / feat_std
        return out

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs = tf.cast(inputs, self.compute_dtype)
        original_shape = tf.shape(inputs)
        return self.instanceNormalize(inputs,original_shape)

def instanceNormalize(feat, eps=1e-5):
    
    n, c = feat.shape[0], 1
    feat_var = tf.math.reduce_variance(tf.reshape(feat,(n, c, -1)), axis=2) + eps
    feat_std = tf.reshape(tf.sqrt(feat_var),(n, c, 1))
    feat_mean = tf.reshape(tf.math.reduce_mean(tf.reshape(feat,(n, c, -1)), axis=2),(n, c, 1))

    out = (feat - feat_mean) / feat_std
    return out

def get_label_test(test_gen):
    test_num = test_gen.samples
    label_test = []
    for i in range((test_num // test_gen.batch_size)+1):
        X,y = test_gen.next()
        label_test.append(y)
            
    label_test = np.argmax(np.vstack(label_test), axis=1)
    print(label_test.shape)
    
    return label_test



















    