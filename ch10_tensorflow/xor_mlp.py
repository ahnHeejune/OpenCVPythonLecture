#  
# to demonstate the model-save-and-restore
# 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import time

# basic XOR example by Tensorflow
def xor_save():

    ##
    ## I. Constrcut Graph
    ##

    # 1. dataset 
    XOR_X = [[0,0],[0,1],[1,0],[1,1]]
    XOR_Y = [[0],[1],[1],[0]]
    # 1.1 input node  (4 is batch size)
    x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
    # 1.2 output node (4 is batch size) 
    y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

    # 2. input to hidden layer 
    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Theta1")
    Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
    # actication function 
    with tf.name_scope("layer2") as scope:
        A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

    # 3. hidden laer to output
    Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Theta2")
    Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")
    # activation function
    with tf.name_scope("layer3") as scope:
        Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

    # 4. Loss or error  (cross-entropy loss)
    with tf.name_scope("cost") as scope:
        cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
            ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
    # 5. optimizer
    with tf.name_scope("train") as scope:
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.initialize_all_variables() 
    
    ##
    ##  2. Run the Graph 
    ##
    # 1. session
    
    sess = tf.Session()
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph_def)
    saver = tf.train.Saver()

    # 2. init  
    sess.run(init)

    t_start = time.time()
    for i in range(200000):  # cost ~ 0.3 at 100,000, cost ~ 0.01 at 200,000
        # bath processing
        # NB: we have only 4 cases here, we use gradient decent not SGD method 
        sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})   
        if i % 10000 == 0:
            #print('-----------------------') 
            #print('Epoch ', i)
            #print('exec ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
            #print('Theta1 ', sess.run(Theta1))
            #print('Bias1 ', sess.run(Bias1))
            #print('Theta2 ', sess.run(Theta2))
            #print('Bias2 ', sess.run(Bias2))
            print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        
        
    print('Fonal parameters-------------') 
    print('Theta1 ', sess.run(Theta1))
    print('Bias1 ', sess.run(Bias1))
    print('Theta2 ', sess.run(Theta2))
    print('Bias2 ', sess.run(Bias2))
    print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
    
    
    t_end = time.time()
    print('Elapsed time ', t_end - t_start)
    
    save_path = saver.save(sess, "./xor/model.ckpt")
    print("Model saved in file: %s" % save_path)
    
    
# basic XOR example by Tensorflow
def xor_restore():

    tf.reset_default_graph() # this is needed for delete graph 
    
    x_ = tf.placeholder(tf.float32, shape=[4,2], name = 'x-input')
    y_ = tf.placeholder(tf.float32, shape=[4,1], name = 'y-input')

    Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name = "Theta1")
    Theta2 = tf.Variable(tf.random_uniform([2,1], -1, 1), name = "Theta2")

    Bias1 = tf.Variable(tf.zeros([2]), name = "Bias1")
    Bias2 = tf.Variable(tf.zeros([1]), name = "Bias2")

    with tf.name_scope("layer2") as scope:
        A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)

    with tf.name_scope("layer3") as scope:
        Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

    with tf.name_scope("cost") as scope:
        cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + 
            ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)

    XOR_X = [[0,0],[0,1],[1,0],[1,1]]
    XOR_Y = [[0],[1],[1],[0]]

    #init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./xor/model.ckpt")
        print("Model restored.")      
        # Check the values of the variables
        print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('Theta1 ', sess.run(Theta1))
        print('Bias1 ', sess.run(Bias1))
        print('Theta2 ', sess.run(Theta2))
        print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        
        # testing modification
        zeroing = Bias1.assign(tf.zeros([2])) 
        print(sess.run(zeroing))
        
        print('Hypothesis ', sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('Theta1 ', sess.run(Theta1))
        print('Bias1 ', sess.run(Bias1))
        print('Theta2 ', sess.run(Theta2))
        print('Bias2 ', sess.run(Bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        


# to demonstate the model-save-and-restore
#    
if __name__  == "__main__":    
    
    #xor_save()      # training model 
    xor_restore()   # use model 
    
    
    