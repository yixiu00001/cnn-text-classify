
# coding: utf-8

# In[1]:

#-*-coding utf-8 -*-


# In[2]:

import tensorflow as tf
import numpy as np


# In[ ]:

class CNNDisease:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate,
                batch_size, decay_steps, decay_rate, sequence_length, vocab_size, embed_size,
                is_training, clip_gradiencets=0.5, decay_rate_big=5.0, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.num_filters_total = self.num_filters*len(filter_sizes)
        
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        
        self.is_training = is_training
        self.clip_gradiencets = clip_gradiencets
        
        #每行输入为词数为sequence_length（实验是21），每次输入行数不定，输出label行数也不定
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")
        #drop选取节点失活的概率
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        #正态分布的张量,tf.random_normal_initializer((mean=0.0, stddev=1.0, seed=None, dtype=tf.float32),stddev 标准差
        self.initializer=initializer
        #初始化embedding weight，输出层的w b
        self.initialize_weights()
        
        if not is_training:
            return
        #预测结果
        self.logits = self.inference()
        self.predicts = tf.argmax(self.logits, 1, name="predictions")
        #计算loss
        self.loss_val = self.loss()
        #计算准确率
        self.accuracy = self.accuracy()
     
        self.train_op = self.train()
        


    def initialize_weights(self):
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable(
                "Embedding", 
                shape=[(self.vocab_size+1), self.embed_size], 
                initializer=self.initializer) 
            self.w_projection = tf.get_variable(
                "w_projection",
                shape=[self.num_filters_total, self.num_classes],
                initializer = self.initializer)
            self.b_projection = tf.get_variable(
                "b_projection",
                shape=[self.num_classes])
    def inference(self):
        #1.get embedding word vector（word2vec trained）
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        self.sentence_embedded_expanded = tf.expand_dims(self.embedded_words, -1)
        #2.loop each filter size conv->relu->pool
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pool-%s" %(filter_size)):
                filter = tf.get_variable(
                    "filter-%s" %(filter_size), 
                    shape=[filter_size, self.embed_size,1, self.num_filters ],
                    initializer = self.initializer)
                conv = tf.nn.conv2d(
                    self.sentence_embedded_expanded,
                    filter,
                    strides=[1,1,1,1],
                    padding = "VALID",
                    name = "conv")
                b = tf.get_variable("b-%s" %(filter_size), shape = [self.num_filters])
                
                h = tf.nn.relu(
                    tf.nn.bias_add(conv, b), 
                    "relu")
                pooled = tf.nn.max_pool(
                    h, 
                    ksize=[1, self.sequence_length-filter_size+1,1,1],
                    strides=[1,1,1,1],
                    padding = "VALID",
                    name = "pool")
                pooled_outputs.append(pooled)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.pooled_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
            
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.pooled_flat, 
                keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.w_projection)+self.b_projection
           
        return logits
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #labels:[batch_size], logits:[batch_size, num_classes]
            #
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y,
                logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name] )* l2_lambda
            
            loss = loss + l2_loss
            return loss
    def accuracy(self):
        prediction_now = tf.equal(tf.cast(self.predicts, tf.int32), self.input_y)
        accuracy = tf.reduce_mean(tf.cast(prediction_now,tf.float32), name="accuracy")
        return accuracy
    def train(self):
        print("learing rate:", self.learning_rate)
        print("global_step:", self.global_step)
        print("decay_steps:", self.decay_steps)
        print("decay_rate:", self.decay_rate)
        print("decay_steps=%d decay_rate=%f" %(self.decay_steps, self.decay_rate))
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, 
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer="Adam",
                clip_gradients=self.clip_gradiencets)
        return train_op

