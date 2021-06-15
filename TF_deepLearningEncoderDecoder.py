import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

#np.random.seed(1)

class deepANNMultiClass:
    def __init__(self,X,layers_dims,layers_dims_dec, optimizer="adam", learning_rate = 0.0075, num_epochs = 3000, print_cost=False, lambd=0,mini_batch_size=400, beta=0,beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
        self.layer_dims=layers_dims
        self.layer_dims_dec=layers_dims_dec
        self.X=X
        #self.Y=self.one_hot_matrix(Y)
        #self.Y=Y
        #self.X_test=X_test
        #self.Y_test=self.one_hot_matrix(Y_test)
        #self.Y_test=Y_test
        self.parameters={}
        self.parameters_dec={}
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
        self.Y_ph=None
        self.X_ph_dec=None
        self.Y_ph_dec=None
        self.last_layer=None
               
    def one_hot_matrix(self,labels):
        """
        Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
        Arguments:
            labels -- vector containing the labels 
            C -- number of classes, the depth of the one hot dimension
    
        Returns: 
            one_hot -- one hot matrix
        """
    
        ### START CODE HERE ###
    
        # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
        C = tf.constant(len(np.unique(labels)), name="C")
    
        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
        # Create the session (approx. 1 line)
        sess = tf.Session()
    
        # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
    
        # Close the session (approx. 1 line). See method 1 above.
        sess.close()
    
        ### END CODE HERE ###
    
        return one_hot
    
    def create_placeholders(self, decoder=False):
        """
        Creates the placeholders for the tensorflow session.
    
        Arguments:
            n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
            n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
        Returns:
        X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
        Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
        Tips:
            - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
              In fact, the number of examples during test/train is different.
        """

        ### START CODE HERE ### (approx. 2 lines)
        if not decoder:
            self.X_ph = tf.placeholder(tf.float32, [self.layer_dims[0],None], name="self.X_ph")
            self.Y_ph = tf.placeholder(tf.float32, [self.layer_dims[-1],None], name="self.Y_ph")
            return self.X_ph, self.Y_ph
        ### END CODE HERE ###
        else:
            self.X_ph_dec = tf.placeholder(tf.float32, [self.layer_dims_dec[0],None], name="self.X_ph_dec")
            self.Y_ph_dec = tf.placeholder(tf.float32, [self.layer_dims_dec[-1],None], name="self.Y_ph_dec")
            return self.X_ph_dec, self.Y_ph_dec
            
      

    
    
    def random_mini_batches(self):
        """
        Creates a list of random minibatches from (X, Y)
    
        Arguments:
            X -- input data, of shape (input size, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
            mini_batch_size -- size of the mini-batches, integer
    
        Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
    
        #np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = self.X.shape[1]                  # number of training examples
        mini_batches = []
        
        # Step 1: Shuffle (X, Y)
        permutation = np.random.permutation(m)
        shuffled_X = self.X[:, permutation]
        #shuffled_Y = self.Y[:, permutation].reshape(self.Y.shape)

        # Step 2: Partition (shuffled_X, shuffled_Y) Minus the end case.
        num_complete_minibatches = math.floor(m/self.mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:,k*self.mini_batch_size:(k+1)*self.mini_batch_size]
            #mini_batch_Y = shuffled_Y[:,k*self.mini_batch_size:(k+1)*self.mini_batch_size]
            ### END CODE HERE ###
            #mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:,math.floor(m/self.mini_batch_size)*self.mini_batch_size:math.floor(m/self.mini_batch_size)*self.mini_batch_size+(m-(math.floor(m/self.mini_batch_size)*self.mini_batch_size))]
            #mini_batch_Y = shuffled_Y[:,math.floor(m/self.mini_batch_size)*self.mini_batch_size:math.floor(m/self.mini_batch_size)*self.mini_batch_size+(m-(math.floor(m/self.mini_batch_size)*self.mini_batch_size))]
        ### END CODE HERE ###
        #mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)
        
        self.mini_batches=mini_batches
    
        return mini_batches


    
       
    def initialize_parameters_deep(self, decoder=False):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
        """

        #np.random.seed(3)
        if not decoder:
            L = len(self.layer_dims)  # number of layers in the network
            print("encode layers",L)
            for l in range(1, L):
                self.parameters['W' + str(l)]=tf.get_variable('W' + str(l), [self.layer_dims[l],self.layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
            #self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                self.parameters['b' + str(l)] = tf.get_variable('b' + str(l), [self.layer_dims[l],1], initializer = tf.zeros_initializer())
            ### END CODE HERE ###
                assert(self.parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
                assert(self.parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
            return self.parameters
        else:
            L = len(self.layer_dims_dec)            # number of layers in the network
            for l in range(1, L):
                self.parameters_dec['W_dec' + str(l)]=tf.get_variable('W_dec' + str(l), [self.layer_dims_dec[l],self.layer_dims_dec[l-1]], initializer = tf.contrib.layers.xavier_initializer())
            #self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                self.parameters_dec['b_dec' + str(l)] = tf.get_variable('b_dec' + str(l), [self.layer_dims_dec[l],1], initializer = tf.zeros_initializer())
            ### END CODE HERE ###
                assert(self.parameters_dec['W_dec' + str(l)].shape == (self.layer_dims_dec[l], self.layer_dims_dec[l-1]))
                assert(self.parameters_dec['b_dec' + str(l)].shape == (self.layer_dims_dec[l], 1))
            
            return self.parameters_dec

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
            A -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
            Z -- the input of the activation function, also called pre-activation parameter
            cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

         
        Z = tf.matmul(W,A)+ b
        #Z = np.dot(W,A) + b
        ### END CODE HERE ###

        #assert(Z.shape == (W.shape[0], A.shape[1]))
        #cache = (A, W, b)

        return Z #cache
    
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
    
    
    def linear_activation_forward(self, A_prev, W, b, activation=None):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
            A -- the output of the activation function, also called the post-activation value
            cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z = self.linear_forward(A_prev,W,b)
            #bn1 = self.batch_norm(Z, True)
            #A = tf.nn.sigmoid(bn1)
            A = tf.nn.sigmoid(Z)
            ### END CODE HERE ###

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z= self.linear_forward(A_prev,W,b)
            bn1=self.batch_norm(Z, True)
            A = tf.nn.relu(bn1)
            #A = tf.nn.relu(Z)
            ### END CODE HERE ###
        elif activation == "tanh":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z= self.linear_forward(A_prev,W,b)
            bn1=self.batch_norm(Z, True)
            A = tf.nn.tanh(bn1)
            #A = tf.nn.tanh(Z)
            ### END CODE HERE ###
        elif activation =="softmax":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            ### START CODE HERE ### (≈ 2 lines of code)
            Z= self.linear_forward(A_prev,W,b)
            #bn1=self.batch_norm(Z, True)
            #A = tf.nn.sigmoid(bn1)
            A = tf.nn.softmax(Z)
            ### END CODE HERE ###
        elif activation == "elu":
            Z= self.linear_forward(A_prev,W,b)
            #bn1=self.batch_norm(Z, True)
            #A = tf.nn.sigmoid(bn1)
            A = tf.nn.elu(Z)
            ### END CODE HERE ###
            
        elif activation==None:
            Z= self.linear_forward(A_prev,W,b)
            return Z
            
            

        #assert (A.shape == (W.shape[0], A_prev.shape[1]))
        #cache = (linear_cache, activation_cache)

        return A  # cache


    def L_model_forward(self,A_ph,params,decoder=False):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            parameters -- output of initialize_parameters_deep()

            Returns:
                AL -- last post-activation value
                caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        #caches = []
        A = A_ph
        parameters=params
        L = len(parameters) // 2                  # number of layers in the neural network
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        
        for l in range(1, L):
            A_prev = A
            ### START CODE HERE ### (≈ 2 lines of code)
            if not decoder:
                A = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b'+ str(l)], "tanh")
            else:
                A = self.linear_activation_forward(A_prev, parameters['W_dec' + str(l)], parameters['b_dec'+ str(l)],"tanh")
            #caches.append(cache)
            ### END CODE HERE ###

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        if not decoder:
            AL = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],"sigmoid")
        else:
            AL = self.linear_activation_forward(A, parameters['W_dec' + str(L)], parameters['b_dec' + str(L)],"sigmoid")
        #caches.append(cache)
        ### END CODE HERE ###
        
        #assert(AL.shape == (1,X.shape[1]))

        return AL  # , caches
    #@staticmethod
    def compute_cost(self,A,TrueLebel,cost_formula="softmax_entropy"):
        """
        Computes the cost
    
        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3
    
        Returns:
        cost - Tensor of the cost function
        """
    
        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(A)
        labels = tf.transpose(TrueLebel)
        if cost_formula=="softmax_entropy":
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        elif cost_formula=="sigmoid_entropy":
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
        elif cost_formula=="RMSprob":
            cost=tf.reduce_mean(tf.pow(TrueLebel - A, 2))
        ### END CODE HERE ###
        
        return cost
    
    def optmize(self,algo):
        
        if algo=="adam":
            return tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        elif algo =="sgd":
            return tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)
        elif algo=="RMSprob":
            return tf.train.RMSPropOptimizer(self.learning_rate)
        
    def model(self):
        """
        3-layer neural network model which can be run in different optimizer modes.
    
        Arguments:
            X -- input data, of shape (2, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
            layers_dims -- python list, containing the size of each layer
            learning_rate -- the learning rate, scalar.
            mini_batch_size -- the size of a mini batch
            beta -- Momentum hyperparameter
            beta1 -- Exponential decay hyperparameter for the past gradients estimates 
            beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
            epsilon -- hyperparameter preventing division by zero in Adam updates
            num_epochs -- number of epochs
            print_cost -- True to print the cost every 1000 epochs

        Returns:
            parameters -- python dictionary containing your updated parameters 
        """
        ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        #tf.set_random_seed(1)                             # to keep consistent results
        #seed = 3                                          # to keep consistent results
        (n_x, m) = self.X.shape                          # (n_x: input size, m : number of examples in the train set)
        #n_y = self.Y.shape[1]                            # n_y : output size
        costs = []           
        #L = len(self.layer_dims)             # number of layers in the neural networks
 
        # Create Placeholders of shape (n_x, n_y)
        ### START CODE HERE ### (1 line)
        X_ph, Y_ph=self.create_placeholders()
        X_ph_dec,Y_ph_dec = self.create_placeholders(decoder=True)
        
        # Initialize parameters
        parameters = self.initialize_parameters_deep()
        parameters_dec = self.initialize_parameters_deep(decoder=True)
        
        # forward propagetion encoder
        Z_Enc = self.L_model_forward(self.X_ph,self.parameters)
        print("encode", Z_Enc.get_shape())
        Z_Dec = self.L_model_forward(self.X_ph_dec,self.parameters_dec,decoder=True)
        print("decode",Z_Dec.get_shape())
        #cost
        cost = self.compute_cost(Z_Dec, self.Y_ph_dec, cost_formula="RMSprob")
        optim=self.optmize("RMSprob").minimize(cost)
        
        num_minibatches = int(m / self.mini_batch_size)
        #print(self.X.shape,m,self.mini_batch_size,num_minibatches)
        # Initialize all the variables
        
        init = tf.global_variables_initializer()       
        saver = tf.train.Saver()
        with tf.Session() as sess:
           
            # Run the initialization
            sess.run(init)
            #print("initial paramenter",sess.run(parameters))
            # Do the training loop
            for epoch in range(self.num_epochs):

                epoch_cost = 0.                       # Defines a cost related to an epoch
                
                
                num_minibatches = int(m / self.mini_batch_size) # number of minibatches of size minibatch_size in the train set
                #seed = seed + 1
                minibatches=self.random_mini_batches()
                #print(len(minibatches))
                #minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)
                
                epoch_encode=np.array([])
                epoch_decode=np.array([])
                    
                for minibatch in minibatches:
                  
                    # Select a minibatch
                    #(minibatch_X, minibatch_Y) = minibatch
                    minibatch_X = minibatch
                    #print(minibatch_X.shape)
                    #print(minibatch_X[0:5,1])
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    encoded=sess.run(Z_Enc, feed_dict={self.X_ph: minibatch_X})
                    decoded=sess.run(Z_Dec,feed_dict={self.X_ph_dec:encoded})
                    
                    _ , minibatch_cost = sess.run([optim, cost], feed_dict={self.X_ph_dec:encoded, self.Y_ph_dec: minibatch_X})
                    
                    
                    #print(minibatch_cost)
                    ### END CODE HERE ###
                
                    epoch_cost += minibatch_cost
                    epoch_encode=np.hstack([epoch_encode, encoded]) if epoch_encode.size else encoded
                    epoch_decode=np.hstack([epoch_decode, decoded]) if epoch_decode.size else decoded
                    #print(encoded.shape, decoded.shape, self.X.shape)
                    
                epoch_cost /=num_minibatches
                # Print the cost every epoch
                if self.print_cost == True and epoch % 1 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if self.print_cost == True and epoch % 1 == 0:
                    costs.append(epoch_cost)
                    
                #costs.append(epoch_cost)
            self.costs=costs
            # plot the cost
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('epochs (per 1)')
            plt.title("Learning rate = " + str(self.learning_rate))
            plt.show()
            
            # lets save the parameters in a variable
            params = sess.run(self.parameters)
            #self.parameters = parameters
            print ("Parameters have been trained!")
            save_path = saver.save(sess,"./model_Encoder_Decoder.ckpt")
            print("Model saved in file: %s" % save_path)
            sess.close()
            
            return epoch_encode, epoch_decode
    
    #test123
    def predict(self,dataset_x):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
            parameters -- python dictionary containing your parameters
            X -- input data of size (n_x, m)

        Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        tf.reset_default_graph()
        X_ph, Y_ph=self.create_placeholders()
        X_ph_dec,Y_ph_dec = self.create_placeholders(decoder=True)
        
        
        parameters = self.initialize_parameters_deep()
        parameters_dec = self.initialize_parameters_deep(decoder=True)
        
        Z_enc=self.L_model_forward(dataset_x, self.parameters)
        Z_dec=self.L_model_forward(Z_enc, self.parameters_dec,decoder=True)
        
        #init = tf.global_variables_initializer()       
        saver = tf.train.Saver() 
        with tf.Session() as sess:
            saver.restore(sess,"./model_Encoder_Decoder.ckpt") 
            # Run the initialization
            #sess.run(init)
            #Calculate the correct predictions
            encode=sess.run(Z_enc, feed_dict={self.X_ph:dataset_x})
            decode=sess.run(Z_dec,feed_dict={self.X_ph_dec:encode})
            sess.close()
            return encode, decode

    def accuracyy(self, TrueLabeles):
        
        correct_prediction = np.sum(np.equal(np.argmax(self.predictions, axis=0), TrueLabeles))
        accuracy = correct_prediction/len(TrueLabeles)
        self.accuracy=accuracy
        return self.accuracy
    def contengencyTable(self):
        return
