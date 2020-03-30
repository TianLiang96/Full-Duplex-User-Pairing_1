import LTE as lte
from RL import *
import tensorflow as tf
import numpy as np
import csv

f = open('reward.csv','w',encoding='utf-8',newline='')
csv_writer = csv.writer(f)

#the parameter of network
global input_num
input_num = 34
output_num = 8

#the parameter of playing
episodes_num = 10015 #training times

#the parameter of LTE
up_user_num = 6
down_user_num = 8
channel_num = 6
cell_r = 100
global up_x,up_y,down_x,down_y

r_jilu=[]
record=np.zeros(32)
length_jilu_2=0
#define an object of DQN class
pair_dqn = DQN(inputnum=input_num,outputnum=output_num)

if __name__=='__main__':
    # play the  game!
    for m in range(episodes_num):
        up_x, up_y, down_x, down_y, state, line = lte.cell_set(up_user_num, down_user_num, cell_r, m%32)  #
        up_x2=up_x.copy()
        up_y2=up_y.copy()
        down_x2=down_x.copy()
        down_y2=down_y.copy()
        juli=jisuan(up_x2,up_y2,down_x2,down_y2)
        action_jilu = np.zeros(up_user_num)  # record the past actions
        if m<28:
            for n in range(up_user_num):
                action = pair_dqn.choose_action(m, n, state, down_user_num,juli)  # get the action
                action_jilu[n] = action  # record the action

                reward = get_reward(up_x2, up_y2, down_x2, down_y2, n, action_jilu)  # get the reward

                state_next = renew_state(line, up_x, up_y, down_x, down_y, action, n, input_num)  # get the next state

                store_memory_expert(state, action, reward, state_next)  # store the memory

                state = state_next
        else:
            for n in range(up_user_num):

                action = pair_dqn.choose_action(m, n, state, down_user_num,juli)  # get the action
                action_jilu[n] = action  # record the action

                reward = get_reward(up_x2, up_y2, down_x2, down_y2, n, action_jilu)  # get the reward

                state_next = renew_state(line, up_x, up_y, down_x, down_y, action, n, input_num)  # get the next state

                store_memory(state, action, reward, state_next)  # store the memory

                state = state_next

                if m > 100:
                    pair_dqn.learn()

        if m % 1000 == 0 :
            pair_dqn.renew_param()

        if m % 1 == 0:
            print(m, reward, action_jilu,m%32)
        if m>=32:
            record[m%32]=reward
            if m%32==31:
                csv_writer.writerow(record)
    f.close()














