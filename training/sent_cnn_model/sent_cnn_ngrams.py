import tensorflow as tf
import numpy as np

class SentCNN(object):
    """
    A CNN for utterance and relation pair matching regression.
    Uses an embedding layer, convolutional layer, max-pooling layer,
    and a logistic regression layer.
    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocabulary_ngrams,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):

        # input_x_u: batch_size x sequence_length
        self.input_x_u = tf.placeholder(tf.float32, 
                                        [None, sequence_length, 1],
                                        name="input_x_u")
        # input_x_r: batch_size x num_classes x sequence_length
        self.input_x_r = tf.placeholder(tf.float32,
                                        [None, num_classes, sequence_length, 1],
                                        name="input_x_r")
        # input_y: batch_size,
        self.input_y = tf.placeholder(tf.int64,
                                      [None],
                                      name="input_y")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        # Store the sequence_length used for the training, needing for test inference.
        self.sequence_length = tf.Variable(sequence_length, trainable=False, dtype=tf.int32, name="sequence_length")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpooling layer for each filter size
        pooled_outputs_u = []
        pooled_outputs_r = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s-u" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)

                conv_u1d = tf.nn.conv1d(
                    self.input_x_u,
                    W,
                    stride=1,
                    padding="VALID",
                    name="conv-u")

                # Apply nonlinearity
                h_u = tf.nn.sigmoid(tf.nn.bias_add(conv_u1d, b), name="activation-u")

                # Maxpooling over outputs
                pooled_u1d = tf.nn.pool(
                    h_u,
                    window_shape=[sequence_length - filter_size + 1],
                    pooling_type="MAX",
                    padding="VALID",
                    strides=[1],
                    name="pool-u")

                pooled_outputs_u.append(pooled_u1d)

                # Pass each element in x_r through the same layer
                pooled_outputs_r_wclasses = []
                for j in range(num_classes):
                    input_x_r_j = self.input_x_r[:, j, :, :]
                    conv_r_j = tf.nn.conv1d(
                        input_x_r_j,
                        W,
                        stride=1,
                        padding="VALID",
                        name="conv-r-%s" % j)

                    h_r_j = tf.nn.sigmoid(tf.nn.bias_add(conv_r_j, b), name="activation-r-%s" % j)

                    pooled_r_j = tf.nn.pool(
                        h_r_j,
                        window_shape=[sequence_length - filter_size + 1],
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
        #self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, 1, num_filters_total])
        self.h_pool_flat_u = self.h_pool_u
        print("DEBUG: h_pool_flat_u -> %s" % self.h_pool_flat_u)


        print("DEBUG: pooled_outputs_r -> %s" % pooled_outputs_r)
        self.h_pool_r = tf.concat(axis=2, values=pooled_outputs_r)
        print("DEBUG: h_pool_r -> %s" % self.h_pool_r)

        # h_pool_flat_r: batch_size x num_classes X num_filters_total
        #self.h_pool_flat_r = tf.reshape(self.h_pool_r, [-1, num_classes, num_filters_total])
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

        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            self.dot =  tf.reduce_sum(tf.multiply(self.h_dropped_u,
                                        self.h_dropped_r), 2)
            print("DEBUG: dot -> %s" % self.dot)
            self.sqrt_u = tf.sqrt(tf.reduce_sum(self.h_dropped_u**2, 2))
            print("DEBUG: sqrt_u -> %s" % self.sqrt_u)
            self.sqrt_r = tf.sqrt(tf.reduce_sum(self.h_dropped_r**2, 2))
            print("DEBUG: sqrt_r -> %s" % self.sqrt_r)
            epsilon = 1e-5
            self.cosine = tf.maximum(self.dot / (tf.maximum(self.sqrt_u * self.sqrt_r, epsilon)), epsilon)
            print("DEBUG: cosine -> %s" % self.cosine)
            self.predictions = tf.argmax(self.cosine, 1, name="predictions")
            print("DEBUG: predictions -> %s" % self.predictions)

        # softmax regression - loss and prediction
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=100*self.cosine, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")