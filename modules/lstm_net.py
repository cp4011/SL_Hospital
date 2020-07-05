import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav
import numpy as np


class LSTM_net:

    def __init__(self, obs_size, nb_hidden=128, action_size=7):

        self.obs_size = obs_size        # 365
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        def __graph__():
            tf.reset_default_graph()

            # entry points
            features_ = tf.placeholder(tf.float32, [1, obs_size], name='input_features')    # 365
            init_state_c_, init_state_h_ = (tf.placeholder(tf.float32, [1, nb_hidden]) for _ in range(2))   # 128
            action_ = tf.placeholder(tf.int32, name='ground_truth_action')  # label
            action_mask_ = tf.placeholder(tf.float32, [action_size], name='action_mask')    # 7个二进制（将要与softmax后的分布相乘）

            # input projection
            Wi = tf.get_variable('Wi', [obs_size, nb_hidden], initializer=xav())    # [365,128]
            bi = tf.get_variable('bi', [nb_hidden], initializer=tf.constant_initializer(0.))    # 128

            # add relu/tanh here if necessary
            projected_features = tf.matmul(features_, Wi) + bi                  # 128

            # state_is_tuple如果为True，则接受和返回的状态是c_state和m_state的2-tuple；如果为False，则他们沿着列轴连接后一种即将被弃用
            lstm_f = tf.contrib.rnn.LSTMCell(nb_hidden, state_is_tuple=True)    # 128

            lstm_op, state = lstm_f(inputs=projected_features, state=(init_state_c_, init_state_h_))

            # reshape LSTM's state tuple (2,128) -> (1,256)  (joint h and c)
            state_reshaped = tf.concat(axis=1, values=(state.c, state.h))   # 256

            # output projection
            Wo = tf.get_variable('Wo', [2 * nb_hidden, action_size], initializer=xav())         # [256,7]
            bo = tf.get_variable('bo', [action_size], initializer=tf.constant_initializer(0.))  # 7

            # get logits
            logits = tf.matmul(state_reshaped, Wo) + bo     # 7
            # probabilities
            #  normalization : elemwise multiply with action mask
            probs = tf.multiply(tf.squeeze(tf.nn.softmax(logits)), action_mask_)    # softmax后的概率分布与7个二进制0 1相乘

            # prediction
            prediction = tf.arg_max(probs, dimension=0)     # 取概率最大的action_id作为输出

            # loss  由于有sparse_，labels为一维向量，长度=batch_size，此处长度为1 [action_id]，代表分类结果
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action_)

            # train op
            train_op = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

            # attach symbols to self
            self.loss = loss
            self.prediction = prediction
            self.probs = probs
            self.logits = logits
            self.state = state
            self.train_op = train_op

            # attach placeholders
            self.features_ = features_
            self.init_state_c_ = init_state_c_
            self.init_state_h_ = init_state_h_
            self.action_ = action_
            self.action_mask_ = action_mask_

        # build graph
        __graph__()

        # start a session; attach to self
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.sess = sess
        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)     # 128
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)     # 128

    # forward propagation
    def forward(self, features, action_mask):   # 389   16
        # forward
        probs, prediction, state_c, state_h = self.sess.run([self.probs, self.prediction, self.state.c, self.state.h],
                                                            feed_dict={
                                                                self.features_: features.reshape([1, self.obs_size]),
                                                                self.init_state_c_: self.init_state_c,
                                                                self.init_state_h_: self.init_state_h,
                                                                self.action_mask_: action_mask
                                                            })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        # return argmax
        return prediction

    # training
    def train_step(self, features, action, action_mask):    # 389   action_id 1bit  16
        _, loss_value, state_c, state_h = self.sess.run([self.train_op, self.loss, self.state.c, self.state.h],
                                                        feed_dict={
                                                            self.features_: features.reshape([1, self.obs_size]),
                                                            self.action_: [action],
                                                            self.init_state_c_: self.init_state_c,
                                                            self.init_state_h_: self.init_state_h,
                                                            self.action_mask_: action_mask
                                                        })
        # maintain state
        self.init_state_c = state_c
        self.init_state_h = state_h
        return loss_value

    def reset_state(self):
        # set init state to zeros
        self.init_state_c = np.zeros([1, self.nb_hidden], dtype=np.float32)
        self.init_state_h = np.zeros([1, self.nb_hidden], dtype=np.float32)

    # save session to checkpoint
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'ckpt/SL_Hospital.ckpt', global_step=0)
        print('\n:: saved to ckpt/SL_Hospital.ckpt \n')

    # restore session from checkpoint
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')
