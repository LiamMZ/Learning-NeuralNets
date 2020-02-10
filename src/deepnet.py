import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

'''
input>>weit>>hl activation>>weights>> ... >>output

compare output to intended output>> cost function(cross entropu)
optimization function (optimizer)>>minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#onehot => pixels are on (1) or off (0) in a 28x28 sized array

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

#height x width
x = tf.placeholder('float'[None, 784])
y = tf.placeholder('float'[None,])

