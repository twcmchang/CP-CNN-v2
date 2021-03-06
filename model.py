import os
import time
import numpy as np
import tensorflow as tf

# VGG_MEAN = [123.68, 116.779, 103.939] # [R, G, B]
VGG_MEAN = [103.939, 116.779, 123.68] # [B, G, R]
class VGG16:
    def __init__(self, classes=10, shape=(32,32,3)):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """
        
        self.gamma_var = []
        self.net_shape = []
        
        # input information
        self.H, self.W, self.C = shape
        self.classes = classes
        
        # operation dictionary
        self.prob_dict = {}
        self.loss_dict = {}
        self.accu_dict = {}

        # parameter dictionary
        self.para_dict = {}

    def build(self, vgg16_npy_path, prof_type=None, conv_pre_training=True, fc_pre_training=True):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        start_time = time.time()
        print("build model started")

        if prof_type is None:
            self.prof_type = "all-one"
        else:
            self.prof_type = prof_type

        # load pre-trained weights
        if isinstance(vgg16_npy_path,dict):
            self.data_dict = vgg16_npy_path
            print("parameters loaded")
        else:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print("npy file loaded")

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.y = tf.placeholder(tf.float32, [None, self.classes])
        self.is_train = tf.placeholder(tf.bool)
        
        self.x = self.x/255.0

        # normalize input by VGG_MEAN
        #red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        #assert   red.get_shape().as_list()[1:] == [self.H, self.W, 1]
        #assert green.get_shape().as_list()[1:] == [self.H, self.W, 1]
        #assert  blue.get_shape().as_list()[1:] == [self.H, self.W, 1]

        #self.x = tf.concat(axis=3, values=[
        #      blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #       red - VGG_MEAN[2],
        #])
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        dp={
            'conv1_1':1.00,
            'conv1_2':1.00,
            'conv2_1':1.00,
            'conv2_2':1.00,
            'conv3_1':1.00,
            'conv3_2':1.00,
            'conv3_3':1.00,
            'conv4_1':1.00,
            'conv4_2':1.00,
            'conv4_3':1.00,
            'conv5_1':1.00,
            'conv5_2':1.00,
            'conv5_3':1.00
            }

        # declare and initialize the weights of VGG16
        with tf.variable_scope("VGG16"):
            # weight decay
            self._weight_decay = 0.0
            for k, v in sorted(dp.items()):
                (conv_filter, gamma, beta, bn_mean, bn_variance), conv_bias = self.get_conv_filter(k, conv_pre_training), self.get_bias(k, conv_pre_training)
                self.para_dict[k] = [conv_filter, conv_bias]
                self.para_dict[k+"_gamma"] = gamma
                self.para_dict[k+"_beta"] = beta
                self.para_dict[k+"_bn_mean"] = bn_mean
                self.para_dict[k+"_bn_variance"] = bn_variance
                self.gamma_var.append(self.para_dict[k+"_gamma"])

                # weight decay
                self._weight_decay += tf.nn.l2_loss(conv_filter)+tf.nn.l2_loss(conv_bias)

            if fc_pre_training:
                fc_W, fc_b = self.get_fc_layer('fc_1', fc_pre_training), self.get_bias('fc_1', fc_pre_training)
                self.para_dict['fc_1'] = [fc_W, fc_b]

                self._weight_decay += tf.nn.l2_loss(fc_W) + tf.nn.l2_loss(fc_b)

                fc_W, fc_b = self.get_fc_layer('fc_2', fc_pre_training), self.get_bias('fc_2', fc_pre_training)
                self.para_dict['fc_2'] = [fc_W, fc_b]

                self._weight_decay += tf.nn.l2_loss(fc_W) + tf.nn.l2_loss(fc_b)
            else:
                # the last fully connected layers should be trained
                # user specified fully connected layers
                fc_W = self.get_fc_layer('fc_1', fc_pre_training, shape=(512, 512))
                fc_b = self.get_bias('fc_1', fc_pre_training, shape=(512,))
                self.para_dict['fc_1'] = [fc_W, fc_b]

                self._weight_decay += tf.nn.l2_loss(fc_W) + tf.nn.l2_loss(fc_b)

                fc_W = self.get_fc_layer('fc_2', fc_pre_training, shape=(512, self.classes))
                fc_b = self.get_bias('fc_2', fc_pre_training, shape=(self.classes,))
                self.para_dict['fc_2'] = [fc_W, fc_b]

                self._weight_decay += tf.nn.l2_loss(fc_W) + tf.nn.l2_loss(fc_b)

        print(("build model finished: %ds" % (time.time() - start_time)))

    def sparsity_train(self, l1_gamma=0.001, l1_gamma_diff=0.001, decay=0.0005, keep_prob=0.0):
        
        self._keep_prob = keep_prob
        start_time = time.time()
        with tf.name_scope("var_dp"):
            conv1_1 = self.idp_conv_bn_layer( self.x, "conv1_1")
            conv1_2 = self.idp_conv_bn_layer(conv1_1, "conv1_2")
            pool1 = self.max_pool(conv1_2, 'pool1')

            conv2_1 = self.idp_conv_bn_layer(  pool1, "conv2_1")
            conv2_2 = self.idp_conv_bn_layer(conv2_1, "conv2_2")
            pool2 = self.max_pool(conv2_2, 'pool2')

            conv3_1 = self.idp_conv_bn_layer(  pool2, "conv3_1")
            conv3_2 = self.idp_conv_bn_layer(conv3_1, "conv3_2")
            conv3_3 = self.idp_conv_bn_layer(conv3_2, "conv3_3")
            pool3 = self.max_pool(conv3_3, 'pool3')

            conv4_1 = self.idp_conv_bn_layer(  pool3, "conv4_1")
            conv4_2 = self.idp_conv_bn_layer(conv4_1, "conv4_2")
            conv4_3 = self.idp_conv_bn_layer(conv4_2, "conv4_3")
            pool4   = self.max_pool(conv4_3, 'pool4')

            conv5_1 = self.idp_conv_bn_layer(  pool4, "conv5_1")
            conv5_2 = self.idp_conv_bn_layer(conv5_1, "conv5_2")
            conv5_3 = self.idp_conv_bn_layer(conv5_2, "conv5_3")
            pool5 = self.max_pool(conv5_3, 'pool5')

            pool5 = self.dropout_layer(pool5, keep_prob=self._keep_prob)

            fc_1 = self.fc_layer(pool5, 'fc_1')
            fc_1 = tf.nn.relu(fc_1)
            fc_1 = self.dropout_layer(fc_1, keep_prob=self._keep_prob)

            logits = self.fc_layer(fc_1, 'fc_2')       
            prob = tf.nn.softmax(logits, name="prob")
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            loss = tf.reduce_mean(cross_entropy)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(x=tf.argmax(logits, 1), y=tf.argmax(self.y, 1)),tf.float32))
            
            # gamma l1 regularization
            l1_gamma_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_gamma)
            gamma_l1 = tf.contrib.layers.apply_regularization(l1_gamma_regularizer, self.gamma_var)

            # gamma_diff l1 regularization
            def non_increasing_constraint_axis_0(a):
                return tf.nn.relu(a[1:]-a[:-1])
            gamma_diff_var = [non_increasing_constraint_axis_0(x) for x in self.gamma_var]

            l1_gamma_diff_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_gamma_diff)
            gamma_diff_l1 = tf.contrib.layers.apply_regularization(l1_gamma_diff_regularizer, gamma_diff_var)

            # gamma_range l1 regularization
            # def range_contraint_axis_0(a):
            #     return tf.subtract(a[-1], a[0])
            # gamma_range_var = [range_contraint_axis_0(x) for x in self.gamma_var]
            # l1_gamma_range_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_gamma_diff)
            # gamma_range_l1 = tf.contrib.layers.apply_regularization(l1_gamma_range_regularizer, gamma_range_var)

            # gradient
            # gamma_grad_var = [tf.nn.relu(tf.gradients(loss, x)[0]) for x in self.gamma_var]
            # l2_gamma_grad_regularizer = tf.contrib.layers.l2_regularizer(scale=l1_gamma_diff)
            # gamma_grad_l2 = tf.contrib.layers.apply_regularization(l2_gamma_grad_regularizer, gamma_grad_var)

            self.prob_dict["var_dp"] = prob
            self.loss_dict["var_dp"] = loss + gamma_l1 + gamma_diff_l1 + self._weight_decay * decay
            self.accu_dict["var_dp"] = accuracy
            
            tf.summary.scalar(name="accu_var_dp", tensor=accuracy)
            tf.summary.scalar(name="loss_var_dp", tensor=loss)
        self.summary_op = tf.summary.merge_all()        
        print(("sparsity train operation setup: %ds" % (time.time() - start_time)))
    
    def set_idp_operation(self, dp, decay=0.0002, keep_prob=1.0):
        self._keep_prob = keep_prob
        if type(dp) != list:
            raise ValueError("when block_variational is False, dp must be a list.")
        self.dp = dp 
        print("DP under test:", np.round(self.dp,2))
        start_time = time.time()
        # create operations at every dot product percentages
        for dp_i in dp:
            with tf.name_scope(str(int(dp_i*100))):
                conv1_1 = self.idp_conv_bn_layer( self.x, "conv1_1", dp_i)
                conv1_2 = self.idp_conv_bn_layer(conv1_1, "conv1_2", dp_i)
                pool1 = self.max_pool(conv1_2, 'pool1')

                if dp_i == 1.0:
                    self.net_shape.append(conv1_1.get_shape())
                    self.net_shape.append(pool1.get_shape())

                conv2_1 = self.idp_conv_bn_layer(  pool1, "conv2_1", dp_i)
                conv2_2 = self.idp_conv_bn_layer(conv2_1, "conv2_2", dp_i)
                pool2 = self.max_pool(conv2_2, 'pool2')

                if dp_i == 1.0:
                    self.net_shape.append(conv2_1.get_shape())
                    self.net_shape.append(pool2.get_shape())

                conv3_1 = self.idp_conv_bn_layer(  pool2, "conv3_1", dp_i)
                conv3_2 = self.idp_conv_bn_layer(conv3_1, "conv3_2", dp_i)
                conv3_3 = self.idp_conv_bn_layer(conv3_2, "conv3_3", dp_i)
                pool3 = self.max_pool(conv3_3, 'pool3')

                if dp_i == 1.0:
                    self.net_shape.append(conv3_1.get_shape())
                    self.net_shape.append(conv3_2.get_shape())
                    self.net_shape.append(pool3.get_shape())

                conv4_1 = self.idp_conv_bn_layer(  pool3, "conv4_1", dp_i)
                conv4_2 = self.idp_conv_bn_layer(conv4_1, "conv4_2", dp_i)
                conv4_3 = self.idp_conv_bn_layer(conv4_2, "conv4_3", dp_i)
                pool4 = self.max_pool(conv4_3, 'pool4')
                
                if dp_i == 1.0:
                    self.net_shape.append(conv4_1.get_shape())
                    self.net_shape.append(conv4_2.get_shape())
                    self.net_shape.append(pool4.get_shape())

                conv5_1 = self.idp_conv_bn_layer(  pool4, "conv5_1", dp_i)
                conv5_2 = self.idp_conv_bn_layer(conv5_1, "conv5_2", dp_i)
                conv5_3 = self.idp_conv_bn_layer(conv5_2, "conv5_3", dp_i)
                pool5 = self.max_pool(conv5_3, 'pool5')

                if dp_i == 1.0:
                    self.net_shape.append(conv5_1.get_shape())
                    self.net_shape.append(conv5_2.get_shape())
                    self.net_shape.append(pool5.get_shape())

                fc_1 = self.fc_layer(pool5, 'fc_1')
                fc_1 = self.dropout_layer(fc_1, keep_prob=self._keep_prob)
                fc_1 = tf.nn.relu(fc_1)
                
                logits = self.fc_layer(fc_1, 'fc_2')
                prob = tf.nn.softmax(logits, name="prob")

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
                loss = tf.reduce_mean(cross_entropy)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(x=tf.argmax(logits, 1), y=tf.argmax(self.y, 1)), dtype=tf.float32))

                self.prob_dict[str(int(dp_i*100))] = prob
                self.loss_dict[str(int(dp_i*100))] = loss + self._weight_decay * decay
                self.accu_dict[str(int(dp_i*100))] = accuracy

                tf.summary.scalar(name="accu_at_"+str(int(dp_i*100)), tensor=accuracy)
                tf.summary.scalar(name="loss_at_"+str(int(dp_i*100)), tensor=loss)
        self.summary_op = tf.summary.merge_all()
        print(("Set dp operations finished: %ds" % (time.time() - start_time)))

    def spareness(self, thresh=0.05):
        N_active, N_total = 0,0
        for gamma in self.gamma_var:
            m = tf.cast(tf.less(tf.abs(gamma), thresh), tf.float32)
            n_active = tf.reduce_sum(m)
            n_total  = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
            N_active += n_active
            N_total  += n_total
        return N_active/N_total

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def dropout_layer(self, bottom, keep_prob):
        if self.is_train == True:
            return tf.nn.dropout(bottom, keep_prob=keep_prob)
        else:
            return bottom

    def idp_conv_bn_layer(self, bottom, name, dp=1.0):
        with tf.name_scope(name+str(int(dp*100))):
            with tf.variable_scope("VGG16",reuse=True):
                conv_filter = tf.get_variable(name=name+"_W")
                conv_biases = tf.get_variable(name=name+"_b")
                conv_gamma  = tf.get_variable(name=name+"_gamma")
                moving_mean = tf.get_variable(name=name+'_bn_mean')
                moving_variance = tf.get_variable(name=name+'_bn_variance')
                beta = tf.get_variable(name=name+'_beta')
            H,W,C,O = conv_filter.get_shape().as_list()
            
            # create a mask determined by the dot product percentage
            n1 = int(O * dp)
            n0 = O - n1
            mask = tf.constant(value=np.append(np.ones(n1, dtype='float32'), np.zeros(n0, dtype='float32')), dtype=tf.float32)
            conv_gamma = tf.multiply(conv_gamma, mask)
            beta = tf.multiply(beta, mask)
            
            conv = tf.nn.conv2d(bottom, conv_filter, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, conv_biases)

            from tensorflow.python.training.moving_averages import assign_moving_average
            def mean_var_with_update():
                mean, variance = tf.nn.moments(conv, [0,1,2], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, 0.99),
                                              assign_moving_average(moving_variance, variance, 0.99)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(self.is_train, mean_var_with_update, lambda:(moving_mean, moving_variance))

            conv = tf.nn.batch_normalization(conv, mean, variance, beta, conv_gamma, 1e-05)
            relu = tf.nn.relu(conv)
            
            return relu

    def fc_layer(self, bottom, name):
        with tf.name_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            
            with tf.variable_scope("VGG16",reuse=True):
                weights = tf.get_variable(name=name+"_W")
                biases = tf.get_variable(name=name+"_b")

            # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_filter(self, name, pre_training):
        if pre_training:
            conv_filter = tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
        else:
            conv_filter = tf.get_variable(shape=self.data_dict[name][0].shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)
        
        H,W,C,O = conv_filter.get_shape().as_list()

        if pre_training and name+"_gamma" in self.data_dict.keys(): 
            gamma = tf.get_variable(initializer=self.data_dict[name+"_gamma"], name=name+"_gamma")
        else:
            gamma = tf.get_variable(initializer=self.get_profile(O, self.prof_type), name=name+"_gamma")

        if pre_training and name+"_beta" in self.data_dict.keys(): 
            beta = tf.get_variable(initializer=self.data_dict[name+"_beta"], name=name+"_beta", trainable=False)
        else:
            beta = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_beta')

        if pre_training and name+"_bn_mean" in self.data_dict.keys(): 
            bn_mean = tf.get_variable(initializer=self.data_dict[name+"_bn_mean"], name=name+"_bn_mean", trainable=False)
        else:
            bn_mean = tf.get_variable(shape=(O,), initializer=tf.zeros_initializer(), name=name+'_bn_mean', trainable=False)

        if pre_training and name+"_bn_variance" in self.data_dict.keys(): 
            bn_variance = tf.get_variable(initializer=self.data_dict[name+"_bn_variance"], name=name+"_bn_variance", trainable=False)
        else:
            bn_variance = tf.get_variable(shape=(O,),initializer=tf.ones_initializer(), name=name+'_bn_variance', trainable=False)
        return conv_filter, gamma, beta, bn_mean, bn_variance

    def get_fc_layer(self, name, pre_training, shape=None):
        if pre_training:
            return tf.get_variable(initializer=self.data_dict[name][0], name=name+"_W")
        else:
            return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_W", dtype=tf.float32)

    def get_bias(self, name, pre_training, shape=None):
        if pre_training:
            bias = tf.get_variable(initializer=self.data_dict[name][1], name=name+"_b")
        else:
            bias = tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name=name+"_b", dtype=tf.float32)
        return bias

    def get_profile(self, C, prof_type):
        def half_exp(n, k=1, dtype='float32'):
            n_ones = int(n/2)
            n_other = n - n_ones
            return np.append(np.ones(n_ones, dtype=dtype), np.exp((1-k)*np.arange(n_other), dtype=dtype))
        if prof_type == "linear":
            profile = np.linspace(2.0,0.0, num=C, endpoint=False, dtype='float32')
        elif prof_type == "all-one":
            profile = np.ones(C, dtype='float32')
        elif prof_type == "half-exp":
            profile = half_exp(C, 2.0)
        elif prof_type == "harmonic":
            profile = np.array(1.0/(np.arange(C)+1), dtype='float32')
        else:
            raise ValueError("prof_type must be \"all-one\", \"half-exp\", \"harmonic\" or \"linear\".")
        return profile
                
