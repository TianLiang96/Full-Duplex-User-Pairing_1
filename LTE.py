#######setting for cell model############
import numpy as np
import math
import matplotlib.pyplot as plt
global line

lte_input_num = 34
####parame setting
# up_user_num = 5
# down_user_num = 6
# channel_num = 4
# cell_r = 100
#up_u_num the number of the uplink users   cell_r:the length of the cell  f:decision to use which seed
def cell_set(up_u_num,down_u_num,cell_r,f):
    global input_num
    line = np.zeros(up_u_num)


    np.random.seed(f+1)
    up_x = np.random.randint(-cell_r, cell_r, (1, up_u_num))
    up_y = np.random.randint(-cell_r, cell_r, (1, up_u_num))
    down_x = np.random.randint(-cell_r, cell_r, (1, down_u_num))
    down_y = np.random.randint(-cell_r, cell_r, (1, down_u_num))
    up_x = up_x.reshape(up_u_num, )
    up_y = up_y.reshape(up_u_num, )
    down_x = down_x.reshape(down_u_num, )
    down_y = down_y.reshape(down_u_num, )
    state = np.hstack((line, up_x, up_y, down_x, down_y))  # (40,)
    state = np.array(state)
    state = state.reshape(1, lte_input_num)
    return up_x,up_y,down_x,down_y,state,line



# if __name__=="__main__":
#     up_x,up_y,down_x,down_y=cell_set(up_user_num,down_user_num,cell_r)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.scatter(up_x,up_y)
#     ax1.scatter(down_x,down_y)
#     plt.show()


