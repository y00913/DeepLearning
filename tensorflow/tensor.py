import tensorflow as tf

텐서 = tf.constant([3,4,5], tf.float32)
텐서2 = tf.constant([6,7,8])

텐서3 = tf.constant([[1,2],[3,4]])

print(tf.add(텐서, 텐서2))
print(tf.subtract(텐서, 텐서2))
print(tf.divide(텐서, 텐서2))
print(tf.multiply(텐서, 텐서2))
print(tf.matmul(텐서, 텐서2))

텐서4 = tf.zeros([2,2,3])

print(텐서3.shape)

print(tf.cast(텐서, tf.int32))

w = tf.Variable(1.0)
print(w.numpy())
w.assign(2)
print(w)