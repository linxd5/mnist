
from inception.slim import ops, scopes
from inception.slim.inception_lindayong import inception
import tensorflow as tf

def inference(images):
    with scopes.arg_scope([ops.conv2d, ops.fc], stddev=0.1, bias=0.1, batch_norm_params={}):
        with scopes.arg_scope([ops.conv2d], kernel_size=[3,3], padding='SAME'):
            with scopes.arg_scope([ops.max_pool], kernel_size=[2,2], padding='SAME'):
                net = ops.conv2d(images, num_filters_out=32)
                net = ops.conv2d(net, num_filters_out=32)
                net = ops.max_pool(net)
                net = ops.conv2d(net, num_filters_out=64)
                net = ops.conv2d(net, num_filters_out=64)
                net = ops.max_pool(net)
                # net = inception(net, 3, [[32], [48,64], [8,16], [3,16]])
                # net = inception(net, 3, [[32], [48,64], [8,16], [3,16]])
                net = ops.conv2d(net, num_filters_out=128)
                net = ops.conv2d(net, num_filters_out=128)
                # net = ops.max_pool(net)
                net = ops.avg_pool(net, [7,7], [1,1])
                net = ops.flatten(net)
                # net = ops.fc(net, num_units_out=1024)
                # net = tf.nn.dropout(net, keep_prob=keep_prob)
                y_conv = ops.fc(net, num_units_out=10, activation=tf.nn.softmax)

                return y_conv

