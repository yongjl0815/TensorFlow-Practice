#!/usr/bin/python3                                                                          

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import sample data                                                                         
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#set limitations/size                                                                       
n_layer1 = 300 #node per layer                                                              
n_layer2 = 300
n_layer3 = 300

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784]) #height x width                                    
y = tf.placeholder('float')

#building neural network model                                                              
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_layer1])),
                      'biases':tf.Variable(tf.random_normal([n_layer1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_layer1, n_layer2])),
                      'biases':tf.Variable(tf.random_normal([n_layer2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_layer2, n_layer3])),
                      'biases':tf.Variable(tf.random_normal([n_layer3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_layer3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    #calculating (input data * weights) + biases                                            
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) #put it thru activation function                                    

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

#train neural network                                                                       
def train_neural_network(x):
    prediction = neural_network_model(x)

    #calculate how wrong it is, need to minimize this                                       
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, label\
s=y) )

    #help minimize cost                                                                     
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #decide how many times it will flow forth and back                                      
    epochs = 5

    with tf.Session() as sess: #do not need to close session when using with                

        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)

        #correctness of predictions                                                         
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        #accuracy                                                                           
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)