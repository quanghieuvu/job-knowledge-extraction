import tensorflow as tf
import numpy as np

class SentCNN(object):
    """
    A CNN for utterance and relation pair matching regression.
    Uses an embedding layer, convolutional layer, max-pooling layer,
    and a logistic regression layer.
    """
    
    def __init__(self, 
                 doc_sequence_length,
                 query_sequence_length,
                 num_classes, 
                 init_embeddings, 
                 filter_sizes, 
                 num_filters,
                 batch_size, # only need this for dropout layer
                 embeddings_trainable=False,
                 l2_reg_lambda=0.0):
        """
        :param doc_sequence_length: The length of our sentences. Here we always pad
        our sentences to have the same length (depending on the longest sentences
        in our dataset).
        :param num_classes: Number of classes in the output layer.
        :param init_embeddings: Pre-trained word embeddings or initialied values.
        :filter_sizes: The number of words we want our convolutional filters to cover. 
        We will have num_filters for each size specified here. For example, [3, 4, 5] 
        means that we will have filters that slide over 3, 4 and 5 words respectively, 
        for a total of 3 * num_filters filters.
        :num_filters: The number of filters per filter size (see above).
        :embeddings_trainable: Train embeddings or not.
        """
        # Placeholders for input, output and dropout

        # input_x_u: batch_size x doc_sequence_length
        self.input_x_u = tf.placeholder(tf.int32, 
                                        [None, doc_sequence_length],
                                        name="input_x_u")
        # input_x_r: batch_size x num_classes x doc_sequence_length
        self.input_x_r = tf.placeholder(tf.int32, 
                                        [None, num_classes, query_sequence_length],
                                        name="input_x_r")
        # input_y: batch_size, 
        self.input_y = tf.placeholder(tf.int64, 
                                      [None],
                                      name="input_y")
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.embedding_size = np.shape(init_embeddings)[1]

        # Store the doc_sequence_length and query_sequence_length used for the training, needed for test inference.
        self.doc_sequence_length = tf.Variable(doc_sequence_length, trainable=False, dtype=tf.int32, name="doc_sequence_length")
        self.query_sequence_length = tf.Variable(query_sequence_length, trainable=False, dtype=tf.int32, name="query_sequence_length")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(init_embeddings,
                            trainable=embeddings_trainable,
                            dtype=tf.float32,
                            name='W')

            # batch_size x doc_sequence_length x embedding_size
            self.embedded_u = tf.nn.embedding_lookup(W, self.input_x_u)
            print("DEBUG: embedded_u -> %s" % self.embedded_u)

            # batch_size x num_classes x doc_sequence_length x embedding_size
            self.embedded_r = tf.nn.embedding_lookup(W, self.input_x_r)
            print("DEBUG: embedded_r -> %s" % self.embedded_r)

        # Create a convolution + maxpooling layer for each filter size
        pooled_outputs_u = []
        pooled_outputs_r = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s-u" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, self.embedding_size, num_filters]
                W_u = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_u')
                b_u = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_u')
                W_r = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W_r')
                b_r = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b_r')
                l2_loss += tf.nn.l2_loss(W_u)
                l2_loss += tf.nn.l2_loss(b_u)
                l2_loss += tf.nn.l2_loss(W_r)
                l2_loss += tf.nn.l2_loss(b_r)

                conv_u1d = tf.nn.conv1d(
                    self.embedded_u,
                    W_u,
                    stride=1,
                    padding="VALID",
                    name="conv-u")

                # Apply nonlinearity
                h_u = tf.nn.tanh(tf.nn.bias_add(conv_u1d, b_u), name="activation-u")

                # Maxpooling over outputs
                pooled_u1d = tf.nn.pool(
                    h_u,
                    window_shape=[doc_sequence_length - filter_size + 1],
                    pooling_type="MAX",
                    padding="VALID",
                    strides=[1],
                    name="pool-u")

                pooled_outputs_u.append(pooled_u1d)
                
                # Pass each element in x_r through the same layer
                pooled_outputs_r_wclasses = []
                for j in range(num_classes):
                    embedded_r_j = self.embedded_r[:, j, :, :]
                    conv_r_j = tf.nn.conv1d(
                        embedded_r_j,
                        W_r, 
                        stride=1,
                        padding="VALID",
                        name="conv-r-%s" % j)
                    
                    h_r_j = tf.nn.sigmoid(tf.nn.bias_add(conv_r_j, b_r), name="activation-r-%s" % j)
                    
                    pooled_r_j = tf.nn.pool(
                        h_r_j,
                        window_shape=[query_sequence_length - filter_size + 1],
                        pooling_type="MAX",
                        strides=[1],
                        padding="VALID",
                        name="pool-r-%s" % j)
                    pooled_outputs_r_wclasses.append(pooled_r_j)

                # out_tensor: batch_size x num_class x num_filters
                out_tensor = tf.concat(axis=1, values=pooled_outputs_r_wclasses)
                pooled_outputs_r.append(out_tensor)
                    
        
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print("DEBUG: pooled_outputs_u -> %s" % pooled_outputs_u)
        self.h_pool_u = tf.concat(axis=2, values=pooled_outputs_u)
        print("DEBUG: h_pool_u -> %s" % self.h_pool_u)
        # batch_size x 1 x num_filters_total
        self.h_pool_flat_u = self.h_pool_u
        print("DEBUG: h_pool_flat_u -> %s" % self.h_pool_flat_u)
        
        
        print("DEBUG: pooled_outputs_r -> %s" % pooled_outputs_r)
        self.h_pool_r = tf.concat(axis=2, values=pooled_outputs_r)
        print("DEBUG: h_pool_r -> %s" % self.h_pool_r)

        # h_pool_flat_r: batch_size x num_classes X num_filters_total
        self.h_pool_flat_r = self.h_pool_r
        print("DEBUG: h_pool_flat_r -> %s" % self.h_pool_flat_r)
        
        # Add dropout layer to avoid overfitting
        with tf.name_scope("dropout"):
            self.h_features = tf.concat(axis=1, values=[self.h_pool_flat_u, self.h_pool_flat_r])
            print("DEBUG: h_features -> %s" % self.h_features)
            self.h_features_dropped = tf.nn.dropout(self.h_features, 
                                                    self.dropout_keep_prob, 
                                                    noise_shape=[tf.shape(self.h_pool_flat_r)[0], 1, num_filters_total])

            self.h_dropped_u = self.h_features_dropped[:, :1, :]
            self.h_dropped_r = self.h_features_dropped[:, 1:, :]

        # Final Fully Connected Layer
        with tf.name_scope("final_fully_connected"):
            self.fc_final_u = tf.contrib.layers.fully_connected(inputs=self.h_dropped_u, num_outputs=128, activation_fn=tf.nn.relu6, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=tf.contrib.layers.xavier_initializer(), trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda), biases_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda))
            self.fc_final_r = tf.contrib.layers.fully_connected(inputs=self.h_dropped_r, num_outputs=128, activation_fn=tf.nn.relu6, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=tf.contrib.layers.xavier_initializer(), trainable=True, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda), biases_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda))
        
        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            self.dot =  tf.reduce_sum(tf.multiply(self.fc_final_u,
                                        self.fc_final_r), 2)
            print("DEBUG: dot -> %s" % self.dot)
            self.sqrt_u = tf.sqrt(tf.reduce_sum(self.fc_final_u**2, 2))
            print("DEBUG: sqrt_u -> %s" % self.sqrt_u)
            self.sqrt_r = tf.sqrt(tf.reduce_sum(self.fc_final_r**2, 2))
            print("DEBUG: sqrt_r -> %s" % self.sqrt_r)
            epsilon = 1e-5
            self.cosine = tf.maximum(self.dot / (tf.maximum(self.sqrt_u * self.sqrt_r, epsilon)), epsilon)
            print("DEBUG: cosine -> %s" % self.cosine)
            self.predictions = tf.argmax(self.cosine, 1, name="predictions")
            print("DEBUG: predictions -> %s" % self.predictions)
        
        # softmax regression - loss and prediction
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=100*self.cosine, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss / 2 + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            
        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")