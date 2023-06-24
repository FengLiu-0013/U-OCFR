#encoding:utf-8
import numpy as np

import tensorflow as tf
import os
def weight_variable(shape,name='weights'):#weight_init
    return tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=0.02))
def bias_variable(shape,name='biases'):#bias_init
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.01))


def AttNet(x,train_bool): 
    def lrelu(x, leak=0.2, name="LeakyRelu"):
         with tf.variable_scope(name):
             f1 = 0.5 * (1 + leak)
             f2 = 0.5 * (1 - leak)
             return f1 * x + f2 * tf.abs(x)
    def conv_residual_shortcut(input,residual,name):
        input_shape = input.get_shape().as_list()
        residual_shape = residual.get_shape().as_list()
        stride_width = int(round(input_shape[1]/residual_shape[1]))
        stride_height = int(round(input_shape[2]/residual_shape[2]))
        equal_channels = input_shape[3]==residual_shape[3]
        short_cut = input 

        W_conv_residual = weight_variable([1,1,input_shape[3],residual_shape[3]],name=name)
        if stride_width >1 or stride_height >1 or not equal_channels:
            shortcut  = tf.nn.conv2d(input,W_conv_residual,strides=[1,stride_width,stride_height,1],padding='SAME')
        return tf.nn.leaky_relu(tf.add(shortcut,residual))
    def block_res(x,filter_input,filter_output,name,strides=1):
        with tf.variable_scope(name):

            w_conv1 = weight_variable([3,3,filter_input,filter_output],name='W_CONV1')
            b_conv1 = bias_variable([filter_output],name='B_CONV1')
            w_conv2 = weight_variable([3,3,filter_output,filter_output],name='W_CONV2')
            b_conv2 = bias_variable([filter_output],name='B_CONV2')
            w_conv3 = weight_variable([3,3,filter_output,filter_output],name='W_CONV3')
            b_conv3 = bias_variable([filter_output],name='B_CONV3')
            w_conv4 = weight_variable([3,3,filter_output,filter_output],name='W_CONV4')
            b_conv4 = bias_variable([filter_output],name='B_CONV4')

            tensor = tf.nn.leaky_relu(tf.nn.atrous_conv2d(x,w_conv1,[1,1],padding='SAME')+b_conv1)
            tensor = tf.nn.leaky_relu(tf.nn.atrous_conv2d(tensor,w_conv2,[2,2],padding='SAME')+b_conv2)
            tensor = tf.nn.leaky_relu(tf.nn.atrous_conv2d(tensor,w_conv3,[5,5],padding='SAME')+b_conv3)
            tensor = tf.nn.leaky_relu(tf.nn.conv2d(tensor,w_conv4,[1,strides,strides,1],padding='SAME')+b_conv4)

            tensor = conv_residual_shortcut(x,tensor,name='shortcut')
            tensor = tf.layers.batch_normalization(tensor,training=train_bool,name='FEN_output')
        return tensor
    def att(x1,x2):
        x2 = tf.nn.softmax(x2)
        return x1 + x1 * x2
    def AtrousSpatialPyramidPoolingModule(x,filter_in,filter_out,name):
        with tf.variable_scope(name):
            concat_list = []
            #ImagePooling AVG
            feature_map_size = tf.shape(x)
            image_features = tf.reduce_mean(x, [1, 2], keepdims=True)
            w_conv1 = weight_variable([1,1,filter_in,128],name='W_CONV1')
            b_conv1 = bias_variable([128],name='B_CONV1')
            image_features = tf.nn.leaky_relu(tf.nn.conv2d(image_features,w_conv1,[1,1,1,1],padding='SAME')+b_conv1)
            concat_list.append(tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2])))

            #1x1conv
            w_conv2 = weight_variable([1,1,filter_in,128],name='W_CONV2')
            b_conv2 = bias_variable([128],name='B_CONV2')
            concat_list.append(tf.nn.leaky_relu(tf.nn.atrous_conv2d(x,w_conv1,[1,1],padding='SAME')+b_conv1))

            #3x3 rate=6
            w_conv3 = weight_variable([3,3,filter_in,128],name='W_CONV3')
            b_conv3 = bias_variable([128],name='B_CONV3')
            concat_list.append(tf.nn.leaky_relu(tf.nn.atrous_conv2d(x,w_conv3,[6,6],padding='SAME')+b_conv3))

            #3x3 rate=12
            w_conv4 = weight_variable([3,3,filter_in,128],name='W_CONV4')
            b_conv4 = bias_variable([128],name='B_CONV4')
            concat_list.append(tf.nn.leaky_relu(tf.nn.atrous_conv2d(x,w_conv4,[12,12],padding='SAME')+b_conv4))

            #3x3 rate=18
            w_conv5 = weight_variable([3,3,filter_in,128],name='W_CONV5')
            b_conv5 = bias_variable([128],name='B_CONV5')
            concat_list.append(tf.nn.leaky_relu(tf.nn.atrous_conv2d(x,w_conv5,[18,18],padding='SAME')+b_conv5))

            tensor = tf.concat(concat_list,axis=-1)

            w_conv6 = weight_variable([1,1,5*128,filter_out],name='W_CONV6')
            b_conv6 = bias_variable([filter_out],name='B_CONV6')
            tensor = tf.nn.leaky_relu(tf.nn.conv2d(tensor,w_conv6,[1,1,1,1],padding='SAME')+b_conv6)
        return tensor

    with tf.variable_scope('FEN'):
        tensor1 = block_res(x,9,64,strides=2,name='block1')
        tensor2 = block_res(tensor1,64,128,strides=2,name='block2')
        tensor3 = block_res(tensor2,128,256,strides=2,name='block3')
        tensor5 = block_res(tensor3,256,512,strides=2,name='block5')
        tensor7 = block_res(tensor5,512,512,strides=2,name='block7')

    with tf.variable_scope('auto_encoder'):

        w_conv1 = weight_variable([3,3,512,512],name='W_CONV1')
        b_conv1 = bias_variable([512],name='B_CONV1')
        tensorB = tf.nn.leaky_relu(tf.nn.conv2d(tensor7,w_conv1,[1,1,1,1],padding='SAME')+b_conv1)
        tensorB1 = tf.image.resize_bilinear(tensorB, size=tf.cast([16,16], tf.int32))


        w_conv2 = weight_variable([3,3,512,256],name='W_CONV2')
        b_conv2 = bias_variable([256],name='B_CONV2')
        tensorB1_ = tf.nn.leaky_relu(tf.nn.conv2d(tensorB1,w_conv2,[1,1,1,1],padding='SAME')+b_conv2)
        tensorB2 = tf.image.resize_bilinear(tensorB1_, size=tf.cast([32,32], tf.int32))


        w_conv3 = weight_variable([3,3,256,9],name='W_CONV3')
        b_conv3 = bias_variable([9],name='B_CONV3')
        tensorB2_ = tf.nn.leaky_relu(tf.nn.conv2d(tensorB2,w_conv3,[1,1,1,1],padding='SAME')+b_conv3)
        tensor_image = tf.image.resize_bilinear(tensorB2_, size=tf.cast([256,256], tf.int32))
        print ("tensor_image",tensor_image.shape)

    with tf.variable_scope('prediction'):

        w_conv1 = weight_variable([3,3,512,512],name='W_CONV1')
        b_conv1 = bias_variable([512],name='B_CONV1')
        tensorA = tf.nn.leaky_relu(tf.nn.conv2d(tensor7,w_conv1,[1,1,1,1],padding='SAME')+b_conv1)
        tensorA1 = tf.image.resize_bilinear(tensorA, size=tf.cast([16,16], tf.int32))

        tensorA1_= att(tensorA1,tensorB1)

        w_conv2 = weight_variable([3,3,512,256],name='W_CONV2')
        b_conv2 = bias_variable([256],name='B_CONV2')
        _tensorA1_ = tf.nn.leaky_relu(tf.nn.conv2d(tensorA1_,w_conv2,[1,1,1,1],padding='SAME')+b_conv2)
        tensorA2 = tf.image.resize_bilinear(_tensorA1_, size=tf.cast([32,32], tf.int32))

        #Unet concat
        _tensorA2 = tf.concat([tensor4,tensorA2],axis=-1)
        w_conv3 = weight_variable([3,3,512,256],name='W_CONV3')
        b_conv3 = bias_variable([256],name='B_CONV3')
        tensorA2_ = tf.nn.leaky_relu(tf.nn.conv2d(_tensorA2,w_conv3,[1,1,1,1],padding='SAME')+b_conv3)

        _tensorA2_ = att(tensorA2_,tensorB2) 

        w_conv4 = weight_variable([3,3,256,12],name='W_CONV4')
        b_conv4 = bias_variable([12],name='B_CONV4')
        tensor_A2 = tf.nn.leaky_relu(tf.nn.conv2d(_tensorA2_,w_conv4,[1,1,1,1],padding='SAME')+b_conv4)
        tensor_predict = tf.image.resize_bilinear(tensor_A2, size=tf.cast([256,256], tf.int32),name='train_output')
        print ("tensor_predict",tensor_predict.shape)
    return tensor_image,tensor_predict





if __name__=='__main__':
    x=tf.placeholder(tf.float32,[None,256,256,9])
    y=tf.placeholder(tf.float32,[None,256,256,12])
    keep_prob=tf.placeholder(tf.float32)
    train_bool = tf.placeholder(tf.bool)
    var = AttNet(x,train_bool)
