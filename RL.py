# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf
import LTE as lte
import random
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

os.environ["CUDA_VISIBLE_DEVICES"]="0"



global lr
lr = 0.00001    #learning rate
y=0.99        #[0,1]  the bigger the value ,the more important the furure reward

bachsize = 128
input_num = 34
output_num = 8

up_num=6
down_num=8

#store memory
global memory
dict_memory = {'state': 0,'action': 0,'reward': 0,'next_state':0 }
memory = []
memory_zhuanjia=[]
memory_length= 5000

class DQN:
    def __init__(self,inputnum,outputnum,units_1=512,units_2=512,units_3=512,units_4=512):
        #parameter of the  neuro network
        self.units_1 = units_1    #the number of neuro cell on the layer
        self.units_2 = units_2
        self.units_3 = units_3
        self.units_4 = units_4
        self.input_num = inputnum
        self.output_num = outputnum
        #define the evalDQN's hide layers

        self.eval_input_1 = tf.placeholder(shape=[None,self.input_num],dtype = tf.float32)                 #input numbers
        self.eval_weights_1 = tf.Variable(tf.random_uniform([self.input_num,self.units_1],-0.1,0.1))
        self.eval_bias_1 = tf.Variable(tf.random_uniform([self.units_1,],0,1))

        self.eval_input_2 = tf.nn.relu(tf.matmul(self.eval_input_1,self.eval_weights_1)+self.eval_bias_1)
        self.eval_weights_2 = tf.Variable(tf.random_uniform([self.units_1,self.units_2],-0.1,0.1))
        self.eval_bias_2 = tf.Variable(tf.random_uniform([self.units_2,],0,1))

        # self.eval_input_3 = tf.nn.relu(tf.matmul(self.eval_input_2,self.eval_weights_2)+self.eval_bias_2)
        # self.eval_weights_3=tf.Variable(tf.random_uniform([self.units_2,self.units_3],-0.1,0.1))
        # self.eval_bias_3=tf.Variable(tf.random_uniform([self.units_3,],0,1))
        #don't forget the renew parameters
        self.eval_input_4 = tf.nn.relu(tf.matmul(self.eval_input_2,self.eval_weights_2)+self.eval_bias_2)

        #define the eval network's output
        self.eval_weights_4 = tf.Variable(tf.random_uniform([self.units_4,self.output_num],-0.1,0.1))
        self.eval_bias_4 = tf.Variable(tf.random_normal([self.output_num,],0,1))
        self.eval_output = tf.matmul(self.eval_input_4,self.eval_weights_4)+self.eval_bias_4
        self.eval_predict = tf.argmax(self.eval_output,1)
        #end of definition

        # define the targetDQN
        self.target_input_1 = tf.placeholder(shape=[None, self.input_num], dtype=tf.float32)
        self.target_weights_1 = self.eval_weights_1
        self.target_bias_1 = self.eval_bias_1
        self.target_input_2 = tf.nn.relu(tf.matmul(self.target_input_1, self.target_weights_1) + self.target_bias_1)
        self.target_weights_2 = self.eval_weights_2
        self.target_bias_2 = self.eval_bias_2
        # self.target_input_3 = tf.nn.relu(tf.matmul(self.target_input_2, self.target_weights_2) + self.target_bias_2)
        # self.target_weights_3 = self.eval_weights_3
        # self.target_bias_3 = self.eval_bias_3
        self.target_input_4 = tf.nn.relu(tf.matmul(self.target_input_2, self.target_weights_2) + self.target_bias_2)
        self.target_weights_4 = self.eval_weights_4
        self.target_bias_4 = self.eval_bias_4
        self.target_output = tf.matmul(self.target_input_4, self.target_weights_4) + self.target_bias_4
        self.target_predict = tf.argmax(self.target_output, 1)
        # end of the definition

        #define the loss to train the network
        self.nextQ = tf.placeholder(shape=[None,self.output_num],dtype=tf.float32)   #nextQ=Q_target
        self.loss = tf.reduce_mean(tf.squared_difference(self.nextQ,self.eval_output))# to min the value between nextQ and eval_output
        self.trainer = tf.train.AdamOptimizer(learning_rate= lr).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    #get the loss in the traifunction
    def get_loss(self, input, next_Q):

            return self.sess.run(self.loss, feed_dict={self.eval_input_1: input, self.nextQ: next_Q})

    # train the eval network
    def dqn_train(self, state_, next_Q):

            self.sess.run(self.trainer,feed_dict={self.nextQ: next_Q, self.eval_input_1: state_})

    #get the action from the network
    def get_action(self, state_, symbol):

            if symbol == 'eval':
                return self.sess.run(self.eval_predict, feed_dict={self.eval_input_1: state_})
            else:
                return self.sess.run(self.target_predict, feed_dict={self.target_input_1: state_})



    #get the Q_value of every action
    def get_output(self, state_,symbol):

            if symbol == 'eval':
                return self.sess.run(self.eval_output, feed_dict={self.eval_input_1: state_})
            else:
                return self.sess.run(self.target_output, feed_dict={self.target_input_1: state_})

    # renew the target_network's parameters
    def renew_param(self):
        self.target_weights_1 = self.eval_weights_1
        self.target_bias_1 = self.eval_bias_1
        self.target_weights_2 = self.eval_weights_2
        self.target_bias_2 = self.eval_bias_2
        # self.target_weights_3 = self.eval_weights_3
        # self.target_bias_3 = self.eval_bias_3
        self.target_weights_4 = self.eval_weights_4
        self.target_bias_4= self.eval_bias_4

    # max greedy strategy for choosing action
    def choose_action(self,p, q, state, down_user_num,juli):
        global linshi, suiji, rand_list
        if p<28:
            row_ind, col_ind = linear_sum_assignment(juli,True)
            suiji=1
            linshi=0
            rand_list=col_ind
        else:
            if q == 0:
                suiji = random.random()
                if p < 500:
                    linshi = 0.3
                elif p >= 500 and p < 1000:
                    linshi = 0.8
                else:
                    linshi = 1 #
                rand_list = random.sample(range(0, down_user_num), down_user_num)

        if suiji > linshi:
            action = rand_list[q]
        else:
            action = self.get_action(state, symbol='eval')
            #  to check the Q_table
            # if p>100 and p%10==0:
            #     Q_table = self.get_output(state,symbol='eval')
            #     print(p,state,Q_table)
        return action

    def learn(self):
        renew_memory = random.sample(memory, bachsize)
        for x in memory_zhuanjia:
            renew_memory.append(copy.copy(x))

        renew_state = np.zeros((bachsize+168, input_num))
        renew_target = np.zeros((bachsize+168, output_num))
        for i in range(bachsize+168):
            renew_state[i] = renew_memory[i]['state']
            action_num = self.get_action(renew_memory[i]['next_state'], symbol='eval')
            Q_t_output = self.get_output(renew_memory[i]['next_state'], symbol='target')
            Q_tihuan = Q_t_output[0][action_num]
            Q_target = self.get_output(renew_memory[i]['state'], symbol='eval')
            if renew_state[i][2*up_num-2] == 0:
                Q_target[0][renew_memory[i]['action']] = renew_memory[i]['reward']
            else:
                Q_target[0][renew_memory[i]['action']] = renew_memory[i]['reward'] + \
                                                         y * Q_tihuan
            renew_target[i] = Q_target
        ###############################################################
        self.dqn_train(renew_state, renew_target)


#compute the length of two users
def length(upx, upy, downx, downy):
    return (np.sqrt((upx - downx) ** 2 + (upy - downy) ** 2))


#get the reward of the action
def get_reward(upx, upy, downx, downy, nn, action_jilu):
    length_jilu = np.zeros(len(upx))
    if nn < len(upx)-1 :
        reward1 = 0
    else:
        set_act = set(action_jilu)
        if len(set_act) != len(action_jilu):
            reward1 = -80
        else:
            for p in range(nn + 1):
                length_jilu[p] = length(upx[p], upy[p], downx[int(action_jilu[p])], downy[int(action_jilu[p])])
            reward1 = 0.1 * np.sum(length_jilu)
    return reward1


#get the next state
def renew_state(line,up_x,up_y,down_x,down_y,action,n,input_num):

    line[n] = action + 1
    up_x[n] = 0
    up_y[n] = 0
    down_x[action] = 0
    down_y[action] = 0
    state_next = np.hstack((line, up_x, up_y, down_x, down_y))
    state_next = np.array(state_next)
    state_next = state_next.reshape(1, input_num)
    return state_next

#store the memory
def store_memory(state,action,reward,state_next):
    dict_memory['state'] = state
    dict_memory['action'] = action
    dict_memory['reward'] = reward
    dict_memory['next_state'] = state_next

    if len(memory) > memory_length:
        memory[0:-1] = memory[1:]
        memory[-1] = dict_memory
    else:
        memory.append(copy.copy(dict_memory))  # if you want use the append methord to add the dictionary,please use this form

#store expert memory
def store_memory_expert(state,action,reward,state_next):
    dict_memory['state'] = state
    dict_memory['action'] = action
    dict_memory['reward'] = reward
    dict_memory['next_state'] = state_next
    memory_zhuanjia.append(copy.copy(dict_memory))  # if you want use the append methord to add the dictionary,please use this form

#计算距离
def jisuan(up_x,up_y,down_x,down_y):
    juli=np.zeros((up_num,down_num))
    for q in range(up_num):
        for p in range(down_num):
            length_jilu_2 = length(up_x[q], up_y[q], down_x[p], down_y[p])
            juli[q][p]=length_jilu_2
    return juli


