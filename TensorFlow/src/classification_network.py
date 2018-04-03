'''
Created on Mar 21, 2018

@author: abhishek
'''

import tensorflow as tf
import numpy as np
#from tensorflow.contrib.eager.python.examples import mnist
#from scipy.special._ufuncs import logit
'''
input > weight >hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function(optimizer) > minimize cost (AdamOptimizer ... SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch
'''

#from create_sentiment_featuresets import create_feature_sets_and_labels

import pickle
import os
f = open('Image_preprocess_one.pickle','rb')
#f = open(os.getcwd()+"/../../ClusterPatterns/src/Image_preprocess_one_sample.pickle",'rb')
train_x,train_y,test_x,test_y = pickle.load(f) 
#10 classes, 0-9
'''
one hot means
 0 = [1,0,0,0,0,0,0,0,0,0]
 1 = [0,1,0,0,0,0,0,0,0,0]
 2 = [0,0,1,0,0,0,0,0,0,0]
'''

n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

#n_classes =3
n_classes =8
batch_size =100

#height * width
x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    input_layer = tf.reshape(data,[-1, 495,69, 3])
    
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    # [247,34,32]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 3], strides=2)
    #[123,16,64]
    '''
    flatten the data before feeding it to dense layer
    ''' 
    data = tf.reshape(pool2, [-1, 123*16 * 64])
    # model for each layer input_data * weight + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([123*16*64,n_nodes_h1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_h1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_h2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_h3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h3,n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])
    return output

def train_neural_networ(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
    #learning rate default = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles feed forward + backprop
    hm_epochs = 4
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])   
                _,c = sess.run([optimizer,cost],feed_dict = {x: batch_x,y: batch_y})
                epoch_loss += c
                i = i+1
            print ("Epoch ",epoch, " completed out of ",hm_epochs," loss:",epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print ("Accuracy ",accuracy.eval({x:test_x,y:test_y}))
        
    
train_neural_networ(x)   
