import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

#np.random.seed(1)

class deepANNMultiClass:
    def __init__(self,X, Y,X_test,Y_test,layers_dims, optimizer="adam", learning_rate = 0.0075, num_epochs = 3000, print_cost=False, lambd=0,mini_batch_size=400, beta=0,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        self.layer_dims=layers_dims
        self.X=X
        self.Y=Y#self.one_hot_matrix(Y)
        self.X_test=X_test
        self.Y_test=Y_test#self.one_hot_matrix(Y_test)
        self.parameters={}
        self.param=None
        self.caches=[]
        self.grads={}
        self.costs=[]
        self.accuracy=None
        self.predictions=None
        self.num_epochs=num_epochs
        self.learning_rate=learning_rate
        self.print_cost=print_cost
        self.lambd=lambd
        self.mini_batches = []
        self.mini_batch_size=mini_batch_size
        self.v=None
        self.beta=beta
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.s=None
        self.optimizer=optimizer
        self.t=0
        self.X_ph=None
        self.Y_ph=[]
        #self.Y_0=None
        #self.Y_1=None
        #self.Y_2=None
        self.last_layer=None
       
               
    def one_hot_matrix(self,labels):
    
        # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
        C = tf.constant(len(np.unique(labels)), name="C")
    
        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
        # Create the session (approx. 1 line)
        sess = tf.Session()
    
        # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
    

        sess.close()
    
        return one_hot
    
    def initialize_Y(self):
        one_hot_train=[self.one_hot_matrix(self.Y[i,:]) for i in range(self.Y.shape[0])]
        one_hot_test=[self.one_hot_matrix(self.Y_test[i,:]) for i in range(self.Y_test.shape[0])]
        self.Y=one_hot_train
        self.Y_test=one_hot_test
        return None
    
    def create_placeholders(self):

        self.X_ph = tf.placeholder(tf.float32, [self.X.shape[0],None], name="self.X_ph")
        self.Y_ph=[tf.placeholder(tf.float32, [self.Y[i].shape[0],None], name="self.Y_"+str(i)) for i in range(len(self.Y))]
        #self.Y_0,self.Y_1,self.Y_2= [tf.placeholder(tf.float32, [self.Y[i].shape[0],None], name="self.Y_"+str(i)) for i in range(len(self.Y))]
        
        return self.X_ph, self.Y_ph

    def random_mini_batches(self):
    
        #np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = self.X.shape[1]                  # number of training examples
        mini_batches = []
        
        # Step 1: Shuffle (X, Y)
        permutation = np.random.permutation(m)
        shuffled_X = self.X[:, permutation]
        shuffled_Y = [self.Y[i][:, permutation].reshape(self.Y[i].shape) for i in range(len(self.Y))]

        # Step 2: Partition (shuffled_X, shuffled_Y) Minus the end case.
        num_complete_minibatches = math.floor(m/self.mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            
            mini_batch_X = shuffled_X[:,k*self.mini_batch_size:(k+1)*self.mini_batch_size]
            mini_batch_Y = [shuffled_Y[i][:,k*self.mini_batch_size:(k+1)*self.mini_batch_size] for i in range(len(self.Y))]
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[:,math.floor(m/self.mini_batch_size)*self.mini_batch_size:math.floor(m/self.mini_batch_size)*self.mini_batch_size+(m-(math.floor(m/self.mini_batch_size)*self.mini_batch_size))]
            mini_batch_Y = [shuffled_Y[i][:,math.floor(m/self.mini_batch_size)*self.mini_batch_size:math.floor(m/self.mini_batch_size)*self.mini_batch_size+(m-(math.floor(m/self.mini_batch_size)*self.mini_batch_size))] for i in range(len(self.Y))]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
        self.mini_batches=mini_batches
    
        return mini_batches

    def initialize_parameters_deep(self):
        #np.random.seed(3)

        L = len(self.layer_dims)            # number of layers in the network
        print(L)
        
        for l in range(1, L-1):
            
            self.parameters['W' + str(l)]=tf.get_variable('W' + str(l), [self.layer_dims[l],self.layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
            #self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters['b' + str(l)] = tf.get_variable('b' + str(l), [self.layer_dims[l],1], initializer = tf.zeros_initializer())
            assert(self.parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
            assert(self.parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
            
        #self.parameters['W' + str(l)]
        _=[tf.get_variable('W' + str(l+1)+str(i), [self.Y[i].shape[0],self.layer_dims[l]], initializer = tf.contrib.layers.xavier_initializer()) for i in range(len(self.Y))]
        
        #self.parameters['b' + str(l)] 
        __= [tf.get_variable('b' + str(l+1)+ str(i), [self.Y[i].shape[0],1], initializer = tf.zeros_initializer()) for i in range(len(self.Y))]
        for j in range(len(_)):
            self.parameters['W' + str(l+1)+str(j)]=_[j]
            self.parameters['b' + str(l+1)+str(j)]=__[j]
        
        #self.parameters.update(sum(self.parameters['W' + str(l)], []))
        
        return self.parameters

    def linear_forward(self, A, W, b):
        
        Z = tf.matmul(W,A)+ b
        
        return Z 
    
    def batch_norm(self, inputs, is_training, decay = 0.999, epsilon = 1e-3):

        scale = tf.Variable(tf.ones([tf.transpose(inputs).get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([tf.transpose(inputs).get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([tf.transpose(inputs).get_shape()[-1]]),  trainable=False)
        pop_var = tf.Variable(tf.ones([tf.transpose(inputs).get_shape()[-1]]), trainable=False)
        
        if is_training:
            batch_mean, batch_var = tf.nn.moments(tf.transpose(inputs),[0])
            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.transpose(tf.nn.batch_normalization(tf.transpose(inputs),
                    batch_mean, batch_var, beta, scale, epsilon))
        else:
            return tf.transpose(tf.nn.batch_normalization(tf.transpose(inputs),
            pop_mean, pop_var, beta, scale, epsilon))
    
    
    def linear_activation_forward(self, A_prev, W, b, activation):

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z = self.linear_forward(A_prev,W,b)
            #bn1 = self.batch_norm(Z, True)
            #A = tf.nn.sigmoid(bn1)
            A = tf.nn.sigmoid(Z)
         

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".           
            Z= self.linear_forward(A_prev,W,b)
            bn1=self.batch_norm(Z, True)
            A = tf.nn.relu(bn1)
            #A = tf.nn.relu(Z)
            
        elif activation == "tanh":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".            
            Z= self.linear_forward(A_prev,W,b)
            bn1=self.batch_norm(Z, True)
            A = tf.nn.tanh(bn1)
            #A = tf.nn.tanh(Z)
        elif activation =="softmax":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z= self.linear_forward(A_prev,W,b)
            #bn1=self.batch_norm(Z, True)
            #A = tf.nn.sigmoid(bn1)
            A = tf.nn.softmax(Z)
        elif activation==None:
            Z= self.linear_forward(A_prev,W,b)
            return Z

        return A


    def L_model_forward(self,X,params,Inner_Layer_activation="relu",output_activation=None):
        
        #caches = []
        A = X
        parameters=params
        L =len(self.layer_dims)                   # number of layers in the neural network
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1,L-1):
            A_prev = A
            A = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b'+ str(l)], Inner_Layer_activation)
        AL = [self.linear_activation_forward(A, parameters['W' + str(l+1)+str(i)], parameters['b' + str(l+1)+str(i)], None) for i in range(len(self.Y))]
        return AL 
    
    def addL2regularizationParameters(self,lambd):
        L=len(self.layer_dims)
        mysum=0
        for l in range(1,L-1):
            mysum+=lambd*tf.nn.l2_loss(self.parameters["W"+str(l)])
        lastL=[lambd*tf.nn.l2_loss(self.parameters["W"+str(l+1)+str(i)]) for i in range(len(self.Y))]
        mysum+=sum(lastL)
        return mysum
        
    def compute_cost(self,A, Y, softmax_entropy=False):
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(A)
        labels = tf.transpose(Y)
        if softmax_entropy:
            cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =labels))+ self.addL2regularizationParameters(self.lambd))
        else:
            cost = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))+ self.addL2regularizationParameters(self.lambd))
        return cost
    
    def multi_task_cost(self,AA, YY):
        
        costs=[self.compute_cost(AA[i],YY[i], softmax_entropy=True) for i in range(len(AA))]
        
        return tf.reduce_mean(costs)
    
    def model(self):
        
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        #seed = 3                                          # to keep consistent results
        (n_x, m) = self.X.shape                          # (n_x: input size, m : number of examples in the train set)
        #n_y = self.Y.shape[1]                            # n_y : output size
        costs = []           
        L = len(self.layer_dims)             # number of layers in the neural networks
 
        # Create Placeholders of shape (n_x, n_y)
        Y=self.initialize_Y()
        X_ph, Y_ph = self.create_placeholders()
        # Initialize parameters
        parameters = self.initialize_parameters_deep()
        # forward propagetion
        Z = self.L_model_forward(self.X_ph, self.parameters)
        #print("Z",str(Z),tf.shape(Z), tf.shape(Y_ph))
        #cost = self.compute_cost(Z,softmax_entropy=True)
        cost = self.multi_task_cost(Z,self.Y_ph)
        #cost = -tf.reduce_sum(Y*tf.log(Z))
        #optimization
        #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)
        if self.optimizer=="adam":
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)
        elif self.optimizer=="sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(cost)
            
        num_minibatches = int(m / self.mini_batch_size)
        #print(self.X.shape,m,self.mini_batch_size,num_minibatches)
        # Initialize all the variables
        
        init=tf.global_variables_initializer()  
        saver = tf.train.Saver()       
        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            # Do the training loop
            for epoch in range(self.num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / self.mini_batch_size) # number of minibatches of size minibatch_size in the train set
                #seed = seed + 1
                minibatches=self.random_mini_batches()

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    #print(self.Y_ph[0].shape,minibatch_Y[0].shape)
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={self.X_ph: minibatch_X, self.Y_ph[0]: minibatch_Y[0],self.Y_ph[1]: minibatch_Y[1],self.Y_ph[2]: minibatch_Y[2]})
                    #print(minibatch_cost)
                    epoch_cost += minibatch_cost
                epoch_cost /=num_minibatches
                # Print the cost every epoch
                if self.print_cost == True and epoch % 10 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if self.print_cost == True and epoch % 10 == 0:
                    costs.append(epoch_cost)
            self.costs=costs
            # plot the cost
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('epochs (per 1)')
            plt.title("Learning rate = " + str(self.learning_rate))
            plt.show()
            # lets save the parameters in a variable
            #self.parameters=parameters
            self.param = sess.run(self.parameters)
            print ("Parameters have been trained!")
            correct_prediction = [tf.equal(tf.argmax(Z[i]), tf.argmax(self.Y_ph[i])) for i in range(len(Z))]
            # Calculate accuracy on the test set
            accuracy = [tf.reduce_mean(tf.cast(correct_prediction[j], tf.float32)) for j in range(len(correct_prediction))]   
            train_accuracy=[accuracy[k].eval({self.X_ph: self.X, self.Y_ph[k]: self.Y[k]}) for k in range(len(self.Y))]
            test_accuracy=[accuracy[c].eval({self.X_ph: self.X_test, self.Y_ph[c]: self.Y_test[c]}) for c in range(len(self.Y_test))]
            print ("Train Accuracy:", sum(train_accuracy)/len(train_accuracy))
            print ("Test Accuracy:", sum(test_accuracy)/len(test_accuracy))
            save_path = saver.save(sess,"./model.ckpt")
            print("Model saved in file: %s" % save_path)
            return self.param
        
    def predict(self, X_dataset):
        
        #parameters=self.parameters
        tf.reset_default_graph()
        #Y=self.initialize_Y()
        X_ph, Y_ph = self.create_placeholders()
        parameters = self.initialize_parameters_deep()
        
        Z=self.L_model_forward(self.X_ph,self.parameters,output_activation="softmax")
        
        saver = tf.train.Saver()      
        with tf.Session() as sess:
            saver.restore(sess,"./model.ckpt")
            self.predictions=sess.run(Z,{self.X_ph:X_dataset})
        
            return [np.argmax(self.predictions[i], axis=0) for i in range(len(self.predictions))]

    def accuracyy(self, TrueLabeles): 
        one_hot_TrueLabeles=[self.one_hot_matrix(TrueLabeles[i,:]) for i in range(TrueLabeles.shape[0])]
        tf.reset_default_graph()
        X_ph, Y_ph = self.create_placeholders()
        parameters = self.initialize_parameters_deep()
        correct_prediction = [tf.equal(tf.argmax(self.predictions[i]), tf.argmax(self.Y_ph[i])) for i in range(len(self.predictions))]
        # Calculate accuracy on the test set
        accuracy = [tf.reduce_mean(tf.cast(correct_prediction[j], tf.float32)) for j in range(len(correct_prediction))]   
        saver = tf.train.Saver()      
        with tf.Session() as sess:
            saver.restore(sess,"./model.ckpt")
            train_accuracy=[accuracy[k].eval({self.Y_ph[k]: one_hot_TrueLabeles[k]}) for k in range(len(self.Y))]
            self.accuracy=sess.run(tf.reduce_mean(train_accuracy))
            
        return self.accuracy
    
    def contengencyTable(self):
        return
