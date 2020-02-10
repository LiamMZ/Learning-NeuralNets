import tensorflow as tf 
import os
os.environ['TF_PY_MIN_LOG_LEVEL'] = '3'


x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2) #x1*x2
sess = tf.Session()
print(sess.run(result))
sess.close()