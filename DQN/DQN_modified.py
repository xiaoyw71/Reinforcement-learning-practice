"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
Update:
xiaoyw71,2021.01.22
https://github.com/xiaoyw71/Reinforcement-learning-practice
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque  # 用于Memory
import random

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features= (None, 50, 50, 1), # 输入灰度图像尺寸与维度
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32, # 按CNN训练经验，缩写batch
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.X_shape = n_features # (None, 50, 50, 1) #输入图像尺寸（maze_env为200*200，resize为100*100）
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()        
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        e_params ,t_params =self._build_net()
        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, shape=self.X_shape, name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, shape=self.X_shape, name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        def q_network(X, name_scope):            
            # Initialize layers
            initializer = tf.contrib.layers.variance_scaling_initializer()        
            with tf.variable_scope(name_scope) as scope:         
                # initialize the convolutional layers
                layer_1 = conv2d(X, num_outputs=32, kernel_size=(5,5), stride=4, padding='SAME', weights_initializer=initializer) 
                tf.summary.histogram('layer_1',layer_1)                
                layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
                tf.summary.histogram('layer_2',layer_2)                
                layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
                tf.summary.histogram('layer_3',layer_3)
                
                # Flatten the result of layer_3 before feeding to the fully connected layer
                flat = flatten(layer_3)
       
                fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
                tf.summary.histogram('fc',fc)
                
                output = fully_connected(fc, num_outputs=self.n_actions, activation_fn=None, weights_initializer=initializer)
                tf.summary.histogram('output',output)                        
                params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                
                return params, output

        # we build our Q network, which takes the input X and generates Q values for all the actions in the state
        mainQ, self.q_eval = q_network(self.s, 'mainQ')
        targetQ, self.q_next = q_network(self.s, 'targetQ')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        return mainQ,targetQ


    def store_transition(self, s, a, r, s_):
        # store all the elements
        self.replay_buffer.append((s, a, r, s_))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})  #后加中括号
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        batch_memory = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [data[0] for data in batch_memory]
        action_batch = [data[1] for data in batch_memory]
        reward_batch = [data[2] for data in batch_memory]
        next_state_batch = [data[3] for data in batch_memory]


        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: state_batch, self.a: action_batch, self.r: reward_batch, self.s_: next_state_batch,
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

if __name__ == '__main__':
    DQN = DeepQNetwork(3,4, output_graph=True)