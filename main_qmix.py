import math
import torch.nn as nn
import torch
import numpy as np
from net import RNN
from qmixnet import QMixNet
import codecs
import csv
import matplotlib.pyplot as plt
import random
import time
import os


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    datas = list(map(lambda x: [x], datas))
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_actions=8*3+1
n_agents=3
n_uav=3
N_STATES=24+50+3+20
N_obs=12
n_bs=5
n_channel=10
max_episode=800
max_step=200
s=np.zeros(16)
s_=np.zeros(16)
last_a=[0,0,0]
last_a_=[0,0,0]
action=np.zeros(3)
timelength=40
agent_action=np.zeros(n_agents)
exp_agent=np.zeros((n_agents,N_obs*2+2))
obs=np.zeros((n_agents,N_obs))
obs_=np.zeros((n_agents,N_obs))
s_=np.zeros(N_STATES)
r=0
c1=0.76*1e7
c2=0.045
loss=0

#UAV初始位置
uav_x=np.array([500,1450,2000])
uav_y=np.array([1650,400,2100])

#基站位置
bs_x=np.array([400,500,1400,1500,2100])
bs_y=np.array([400,1600,400,1700,2100])
n_bsusers=np.load('n_bsusers.npy')

bidx0=np.array([0,1,10,11,13,14,15,16,17,18,19])
bidx1 = np.array([2, 3, 45, 46, 47])
bidx2 = np.array([4,5,48,49])
bidx3= np.array([6,7,21,22,23,25])
bidx4 = np.array([8,9])
bsu=np.array([11,5,4,6,2])
btraffic=np.zeros(n_bs)

x1_record=[]
x2_record=[]
x3_record=[]
y1_record=[]
y2_record=[]
y3_record=[]

#用户分布
n_users=50
task=np.zeros(n_users)
users_x=np.load('users_x.npy')#初始用户分布
users_y=np.load('users_y.npy')
score=np.zeros(n_users)
task_score=np.zeros(n_users)


SINRuav=np.zeros((n_uav,n_users))
SINRbs=np.zeros((n_bs,n_users))
loss1=np.zeros(n_users)

Iuav=np.zeros((n_uav,n_users))
Ibs=np.zeros((n_bs,n_users))
ruav=np.zeros((n_agents,n_users))
rbs=np.zeros((n_bs,n_users))
cellstep_record=np.zeros((max_episode*max_step))
Pbscell_record=np.zeros((n_bs,max_episode*max_step))
Puavcell_record=np.zeros((n_uav,max_episode*max_step))
PBS_record=np.zeros((n_bs,max_episode*max_step))
PUAV_record=np.zeros((n_uav,max_episode*max_step))
total_inter=np.zeros(max_episode*max_step)

asso_uavusers=np.zeros((n_uav,n_users)) #uav-users关联矩阵

UAV_M = 20 #无人机质量
rou = 1.225 #thou
A = 0.503  #旋翼面积
P_l = (UAV_M/2*rou*A)**2
P_hover = UAV_M*math.sqrt(UAV_M/(2*rou*A))

P_level=[0.1,0.5,1]
v_level=np.array([10,20,30])
P_uav=np.array([0.1,0.1,0.1]) #初始化功率矩阵
P_bs=np.array([1.0,1.0,1.0,1.0,1.0])
uav_v=np.zeros(n_uav)
P_fly=np.zeros(n_uav)
n_plevel=3

reward_save=np.zeros(max_episode*max_step)
Epi_reward=np.zeros(max_episode)

#网络参数
fre_carrier=3.45e9
MEMORY_CAPACITY=40000
EPSILON=0.9
TARGET_REPLACE_ITER=50
LR=0.00005
gamma=0.9
BATCH_SIZE=256
b_memory=0
rnn_hidden_dim=256

#仿真参数
Pbs_static=500
Puav_static=100
cover_uav=400
cover_bs=250
noisepower=1.0e-10
a=9.61
b=0.16
italos=1
itanlos=100
UAV_h=200
uav_h=80
bs_h=30
gama=4
alpha=3
B=5000000
flag=np.zeros(n_uav)

class QMIX:
    def __init__(self,):
        # 神经网络
        self.learn_step_counter = 0  # for target updating
        self.memory = np.zeros((MEMORY_CAPACITY, n_agents,N_STATES * 2+N_obs*2 + 3))  # 初始记忆大小
        self.eval_rnn = RNN().to(device)  # 每个agent选动作的网络
        self.target_rnn = RNN().to(device)
        self.eval_qmix_net = QMixNet().to(device)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet().to(device)
        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QMIX')

    def learn(self, ):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        if self.learn_step_counter > 0 and self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.learn_step_counter += 1
        b_memory = np.empty((BATCH_SIZE * timelength, 3, N_STATES * 2 + N_obs * 2 + 3))
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        for i in range(BATCH_SIZE):
            if sample_index[i] > MEMORY_CAPACITY - timelength:
                sample_index[i] = sample_index[i] - timelength
            b_memory[i * timelength:i * timelength + timelength, :, :] = self.memory[np.arange(sample_index[i],sample_index[i] + timelength), :]
        mask = np.zeros((BATCH_SIZE * timelength, 1))
        q_evals, q_targets = self.get_q_values(b_memory)

        b_s = b_memory[:, :, 2 * N_obs + 2:2 * N_obs + 2 + N_STATES].reshape(BATCH_SIZE * timelength, n_agents, -1)
        b_s_ = b_memory[:, :, 2 * N_obs + 2 + N_STATES:2 * N_obs + 2 + 2 * N_STATES].reshape(BATCH_SIZE * timelength,n_agents, -1)
        b_r = b_memory[:, :, 2 * N_obs + 2 + 2 * N_STATES:2 * N_obs + 3 + 2 * N_STATES].reshape(BATCH_SIZE * timelength,n_agents, -1)

        b_s = b_s[:, 0, :]
        b_s_ = b_s_[:, 0, :]
        b_r = b_r[:, 0, :]
        for i in range(BATCH_SIZE * timelength):
            if np.sum(b_s[i, :]) != 0:
                mask[i, 0] = 1
        b_r = b_r.reshape(BATCH_SIZE * timelength, 1)
        b_s = torch.tensor(b_s).to(device).float()
        b_s_ = torch.tensor(b_s_).to(device).float()
        b_r = torch.tensor(b_r).to(device).float()
        mask = torch.tensor(mask).to(device)

        q_total_eval = self.eval_qmix_net(q_evals, b_s)#MIX网络拟合全局Q
        q_total_target = self.target_qmix_net(q_targets, b_s_)

        targets = b_r + gamma * q_total_target
        td_error = (q_total_eval - targets.detach())
        mask_td = mask * td_error
        loss = ((mask_td ** 2).sum() / mask.sum()) * 0.5
        print('******************loss*****************************= ', loss)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, max_norm=1e5, norm_type=2)
        self.optimizer.step()


    def choose_action(self, x,xx,h,agent_num):
        x=np.hstack((x,xx))
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(device)                 # 数据维度进行扩充
        h=h.to(device)
        # input only one sample
        if np.random.uniform() <1 - EPSILON:   # greedy
            out = self.eval_rnn.forward(x,h)
            actnum = out[0].argmax().item()
            eval_hstate[agent_num,:]=out[1]
            h=out[1]
        else:   # random
            actnum=np.random.randint(0, n_actions)
            h=h
        return actnum,h

    def store_transition(self, exp):
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY # 返回结果为按顺序堆叠numpy的数组(按列堆叠一个)。
        self.memory[index, :] = exp.copy()
        self.memory_counter += 1

    def get_q_values(self, b_memory):
        b_o = b_memory[:, :, :N_obs]
        b_lasta = b_memory[:, :, N_obs:N_obs + 1].astype(int)
        b_o_ = b_memory[:, :, N_obs + 1:2 * N_obs + 1]
        b_lasta_=b_memory[:, :, 2 * N_obs + 1:2 * N_obs + 2]
        b_o = b_o.reshape(BATCH_SIZE, timelength,n_agents, -1)
        b_lasta = b_lasta.astype(int).reshape(BATCH_SIZE,timelength,n_agents, -1)
        b_lasta_ = b_lasta_.astype(int).reshape(BATCH_SIZE, timelength, n_agents, -1)
        b_o_ = b_o_.reshape(BATCH_SIZE,timelength,n_agents, -1)
        b_o = torch.FloatTensor(b_o).to(device)
        b_lasta = torch.LongTensor(b_lasta).to(device)
        b_lasta_ = torch.LongTensor(b_lasta_).to(device)
        b_o_ = torch.FloatTensor(b_o_).to(device)
        q_evallist=[]
        q_targetlist=[]
        a_index=[]
        for j in range(n_agents):
            agent_b_o=b_o[:,:,j,:]
            agent_lasta=b_lasta[:,:,j,:]
            #agent_b_o=agent_b_o.view((64,10,10))
            agent_b_o_ = b_o_[:,:,j,:]
            agent_lasta_ = b_lasta_[:, :, j, :]
            #agent_b_o_=agent_b_o_.view((64,10,10))
            self.init_hidden_state()
            agent_eval_hidden = self.eval_hidden[:, :, j, :]
            agent_eval_hidden = agent_eval_hidden.view((BATCH_SIZE, rnn_hidden_dim)).to(device)
            agent_target_hidden = self.eval_hidden[:, :, j, :]
            agent_target_hidden = agent_target_hidden.view((BATCH_SIZE, rnn_hidden_dim)).to(device)
            for k in range(timelength):
                o=agent_b_o[:,k,:].to(device)
                o_=agent_b_o_[:,k,:].to(device)
                last_a=agent_lasta[:,k,:].to(device)
                last_a_=agent_lasta_[:,k,:].to(device)
                o=torch.cat((o,last_a),dim=1).to(device)
                o_=torch.cat((o_, last_a_), dim=1).to(device)
                q_eval, agent_eval_hidden = self.eval_rnn(o, agent_eval_hidden)
                q_target, agent_target_hidden = self.target_rnn(o_, agent_target_hidden)
                q_eval = q_eval.data.cpu().numpy()
                q_target = q_target.data.cpu().numpy()
                last_a_ = last_a_.data.cpu().numpy()
                q_evallist.append(q_eval)
                q_targetlist.append(q_target)
                a_index.append(last_a_)
        q_evallist=np.array(q_evallist)
        q_targetlist = np.array(q_targetlist)
        a_index = np.array(a_index)
        qevals=q_evallist.reshape(-1,n_actions)
        qtargets = q_targetlist.reshape(-1, n_actions)
        qevals = torch.tensor(qevals).to(device)
        qtargets  = torch.tensor(qtargets ).to(device)
        a_index=torch.tensor(a_index ).to(device)
        a_index=a_index.view((BATCH_SIZE*n_agents*timelength,1))
        qevals = qevals.gather(0, a_index)
        qevals=qevals.view((BATCH_SIZE*timelength,1,n_agents))
        qtargets = qtargets.max(dim=1)[0]
        qtargets = qtargets.view((BATCH_SIZE*timelength,1, n_agents))
        return qevals, qtargets

    def init_hidden_state(self,):
            self.eval_hidden = torch.zeros(( 1,BATCH_SIZE,n_agents,rnn_hidden_dim))
            self.target_hidden = torch.zeros(( 1,BATCH_SIZE,n_agents, rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')

def taskgenerate():
    for i in range(50):                # 每个时隙数据包数量
        task[i]=random.uniform(1,4)*8
    bt0 = np.sum(task[bidx0])
    bt1 = np.sum(task[bidx1])
    bt2 = np.sum(task[bidx2])
    bt3 = np.sum(task[bidx3])
    bt4 = np.sum(task[bidx4])
    btraffic = np.array([bt0, bt1, bt2, bt3, bt4])
    return btraffic

def obsnext(j,agent_action):
    idx=0
    points = 0
    for i in range(n_users):
        if j == 0:
            d = math.sqrt(((uav_x[0] - users_x[i]) ** 2) + ((uav_y[0] - users_y[i]) ** 2))
            l = np.hstack([uav_x[0], uav_y[0]])
        elif j == 1:
            d = math.sqrt(((uav_x[1]  - users_x[i]) ** 2) + ((uav_y[1]- users_y[i]) ** 2))
            l = np.hstack([uav_x[1], uav_y[1]])
        else:
            d = math.sqrt(((uav_x[2] - users_x[i]) ** 2) + ((uav_x[2] - users_y[i]) ** 2))
            l = np.hstack([uav_x[2], uav_y[2]])
        if d <= cover_uav:
            points = points + (score[i])
            idx = idx + 1
    if idx==0:
        points=1000
    else:
        points = points / idx
    traffic_bs=btraffic
    l0 = np.array([points])
    o_ = np.hstack([l,l0,traffic_bs,agent_action,flag[j]])
    return o_

def state_next(a):
    d1=np.zeros(n_bs)
    d2 = np.zeros(n_bs)
    d3 = np.zeros(n_bs)
    for i in range(n_bs):
        d1[i]=math.sqrt(((bs_x[i]  - uav_x[0]) ** 2) + ((bs_y[i] - uav_y[0]) ** 2))
        d2[i] = math.sqrt(((bs_x[i] - uav_x[1]) ** 2) + ((bs_y[i] - uav_y[1]) ** 2))
        d3[i] = math.sqrt(((bs_x[i] - uav_x[2]) ** 2) + ((bs_y[i] - uav_y[2]) ** 2))
    state0 = np.hstack([uav_x, uav_y, bs_x, bs_y,a,btraffic,score,flag,bsu,d1,d2,d3])
    s_ = state0.reshape(-1)
    return s_

def u_score(score):
    for i in range(n_users):
        flag0 = 0
        flag1 = 0
        d0,d1,d2=math.sqrt(((uav_x[0]-users_x[i])** 2)+((uav_y[0]-users_y[i])**2)),math.sqrt(((uav_x[1]-users_x[i])** 2)+((uav_y[1]-users_y[i])**2)),math.sqrt(((uav_x[2]-users_x[i])** 2)+((uav_y[2]-users_y[i])**2))
        d3,d4,d5=math.sqrt(((bs_x[0]-users_x[i])** 2)+((bs_y[0]-users_y[i])**2)),math.sqrt(((bs_x[1]-users_x[i])** 2)+((bs_y[1]-users_y[i])**2)),math.sqrt(((bs_x[2]-users_x[i])** 2)+((bs_y[2]-users_y[i])**2))
        d6,d7=math.sqrt(((bs_x[3]-users_x[i])** 2)+((bs_y[3]-users_y[i])**2)),math.sqrt(((bs_x[4]-users_x[i])** 2)+((bs_y[4]-users_y[i])**2))
        if d0<= cover_uav or d1<=cover_uav or d2<=cover_uav:
            flag0=1
        if d3<= cover_bs or d4<=cover_bs or d5<=cover_bs or d6<=cover_bs or d7<=cover_bs:
            flag1=1
        if flag0==1 or flag1==1:
            score[i]=score[i]+1
    return score

def reward():
    r1=0
    for i in range(n_users):
        d0,d1,d2=math.sqrt(((uav_x[0]-users_x[i])** 2)+((uav_y[0]-users_y[i])**2)),math.sqrt(((uav_x[1]-users_x[i])** 2)+((uav_y[1]-users_y[i])**2)),math.sqrt(((uav_x[2]-users_x[i])** 2)+((uav_y[2]-users_y[i])**2))
        d3,d4,d5=math.sqrt(((bs_x[0]-users_x[i])** 2)+((bs_y[0]-users_y[i])**2)),math.sqrt(((bs_x[1]-users_x[i])** 2)+((bs_y[1]-users_y[i])**2)),math.sqrt(((bs_x[2]-users_x[i])** 2)+((bs_y[2]-users_y[i])**2))
        d6,d7=math.sqrt(((bs_x[3]-users_x[i])** 2)+((bs_y[3]-users_y[i])**2)),math.sqrt(((bs_x[4]-users_x[i])** 2)+((bs_y[4]-users_y[i])**2))

        if d0<= cover_uav or d1<=cover_uav or d2<=cover_uav:
            flag0=1
        else:
            flag0 = 0
        if d3<= cover_bs or d4<=cover_bs or d5<=cover_bs or d6<=cover_bs or d7<=cover_bs:
            flag1=1
        else:
            flag1 = 0
        if flag0==0 and flag1==0:
            task_score[i]=0.1*(2-score[i]/max_step)
        elif flag0==1 and flag1==0:
            task_score[i]=1.2**(2-score[i]/max_step)
        else:
            task_score[i] = 0
    r1 = task_score * task
    r1=np.sum(r1)
    return r1

def loactionchange(agent_action,uav_x,uav_y):
    if agent_action[0]==24:
        uav_y[0] = uav_y[0]
        uav_x[0]=uav_x[0]
    else:
        a1 = int(agent_action[0] / 3)
        uav_v[0] = v_level[int(agent_action[0] % 3)]
        if a1 == 0:
            uav_y[0] = uav_y[0] + uav_v[0]
        elif a1 == 1:
            uav_x[0] = uav_x[0] + uav_v[0] * math.cos(math.pi / 4)
            uav_y[0] = uav_y[0] + uav_v[0] * math.cos(math.pi / 4)
        elif a1 == 2:
            uav_x[0] = uav_x[0] + uav_v[0]
        elif a1 == 3:
            uav_x[0] = uav_x[0] + uav_v[0] * math.cos(math.pi / 4)
            uav_y[0] = uav_y[0] - uav_v[0] * math.cos(math.pi / 4)
        elif a1 == 4:
            uav_y[0] = uav_y[0] - uav_v[0]
        elif a1 == 5:
            uav_x[0] = uav_x[0] - uav_v[0] * math.cos(math.pi / 4)
            uav_y[0] = uav_y[0] - uav_v[0] * math.cos(math.pi / 4)
        elif a1 == 6:
            uav_x[0] = uav_x[0] - uav_v[0]
        elif a1 == 7:
            uav_x[0] = uav_x[0] - uav_v[0] * math.cos(math.pi / 4)
            uav_y[0] = uav_y[0] + uav_v[0] * math.cos(math.pi / 4)

    if agent_action[1]==24:
        uav_y[1] = uav_y[1]
        uav_x[1] = uav_x[1]
    else:
        a2 = int(agent_action[1] / 3)
        uav_v[1] = v_level[int(agent_action[1] % 3)]
        if a2 == 0:
            uav_y[1] = uav_y[1] + uav_v[1]
        elif a2 == 1:
            uav_x[1] = uav_x[1] + uav_v[1] * math.cos(math.pi / 4)
            uav_y[1] = uav_y[1] + uav_v[1] * math.cos(math.pi / 4)
        elif a2 == 2:
            uav_x[1] = uav_x[1] + uav_v[1]
        elif a2 == 3:
            uav_x[1] = uav_x[1] + uav_v[1] * math.cos(math.pi / 4)
            uav_y[1] = uav_y[1] - uav_v[1] * math.cos(math.pi / 4)
        elif a2 == 4:
            uav_y[1] = uav_y[1] - uav_v[1]
        elif a2 == 5:
            uav_x[1] = uav_x[1] - uav_v[1] * math.cos(math.pi / 4)
            uav_y[1] = uav_y[1] - uav_v[1] * math.cos(math.pi / 4)
        elif a2 == 6:
            uav_x[1] = uav_x[1] - uav_v[1]
        elif a2 == 7:
            uav_x[1] = uav_x[1] - uav_v[1] * math.cos(math.pi / 4)
            uav_y[1] = uav_y[1] + uav_v[1] * math.cos(math.pi / 4)

    if agent_action[2]==24:
        uav_y[2] = uav_y[2]
        uav_x[2] = uav_x[2]
    else:
        a3 = int(agent_action[2] / 3)
        uav_v[2] = v_level[int(agent_action[2] % 3)]
        if a3 == 0:
            uav_y[2] = uav_y[2] + uav_v[2]
        elif a3 == 1:
            uav_x[2] = uav_x[2] + uav_v[2] * math.cos(math.pi / 4)
            uav_y[2] = uav_y[2] + uav_v[2] * math.cos(math.pi / 4)
        elif a3 == 2:
            uav_x[2] = uav_x[2] + uav_v[2]
        elif a3 == 3:
            uav_x[2] = uav_x[2] + uav_v[2] * math.cos(math.pi / 4)
            uav_y[2] = uav_y[2] - uav_v[2] * math.cos(math.pi / 4)
        elif a3 == 4:
            uav_y[2] = uav_y[2] - uav_v[2]
        elif a3 == 5:
            uav_x[2] = uav_x[2] - uav_v[2] * math.cos(math.pi / 4)
            uav_y[2] = uav_y[2] - uav_v[2] * math.cos(math.pi / 4)
        elif a3 == 6:
            uav_x[2] = uav_x[2] - uav_v[2]
        elif a3 == 7:
            uav_x[2] = uav_x[2] - uav_v[2] * math.cos(math.pi / 4)
            uav_y[2] = uav_y[2] + uav_v[2] * math.cos(math.pi / 4)

    return uav_x,uav_y,uav_v

def initial_loc():
    uav_x = np.array([800, 1200, 1500])
    uav_y = np.array([1000,1000, 1000])
    return uav_x,uav_y

def trajectory_record():
    x1_record.append(uav_x[0])
    x2_record.append(uav_x[1])
    x3_record.append(uav_x[2])
    y1_record.append(uav_y[0])
    y2_record.append(uav_y[1])
    y3_record.append(uav_y[2])

def asso_uav_users():
    for j in range(n_uav):
        asso_uavusers[j, :] = 0 # 初始化
        for i in range(n_users):
            d = math.sqrt(((uav_x[j] - users_x[i]) ** 2) + ((uav_y[j] - users_y[i]) ** 2))
            if d <= cover_uav:
                asso_uavusers[j, i] = 1
    return asso_uavusers

def obs_initial():
    obs[:,:]=0
    return obs
def Puav_comnsume(uav_v):
    for i in range(n_uav):
        P_level = UAV_M ** 2 / (math.sqrt(2) * rou * A * math.sqrt((uav_v[i])**2+math.sqrt((((uav_v[i])**2)**2)+P_l)))
        P_drag = 0.125*0.012*rou*A*math.sqrt((uav_v[i])**2)*((uav_v[i])**2)
        P_fly[i] = P_level+P_drag
    return P_fly
def loacation_back():
    if uav_x[0]<0 or uav_x[0]>2600 or uav_y[0]<0 or uav_y[0]>2550:
        uav_x[0]=800
        uav_y[0]=1000
    if uav_x[1]<0 or uav_x[1]>2600 or uav_y[1]<0 or uav_y[1]>2550:
        uav_x[1]=1200
        uav_y[1]=1000
    if uav_x[2]<0 or uav_x[2]>2600 or uav_y[2]<0 or uav_y[2]>2550:
        uav_x[2]=1500
        uav_y[2]=1000
    return uav_x,uav_y

qmix=QMIX()
for episode in range(max_episode):
    uav_x,uav_y=initial_loc()
    score[:]=0
    eval_hstate = torch.zeros([n_agents, 1, rnn_hidden_dim])
    for step in range(max_step):
        t0 = time.time()
        btraffic=taskgenerate()
        s=s_
        for j in range(n_agents):
            if step==0:
                o=obs_initial()   #状态初始化
                obs[j, :]=o[j,:]
            obs[j,:]=obs_[j,:]
            o=obs[j,:]
            agent_action[j],eval_hstate[j,:]=qmix.choose_action(o,last_a[j],eval_hstate[j,:],j)
        uav_x,uav_y,uav_v= loactionchange(agent_action,uav_x,uav_y)
        print('action=', agent_action)
        trajectory_record()
        # 飞行结束后 初始化uav-user关联
        asso_uavusers = asso_uav_users()
        P_fly=Puav_comnsume(uav_v)  #o_作为下一个o
        score = u_score(score)
        for j in range(n_agents):
            obs_[j,:]=obsnext(j,agent_action)
            last_a_[j]=agent_action[j]
            xx=np.hstack((obs[j,:],last_a[j],obs_[j,:],last_a_[j]))
            last_a[j] = agent_action[j]
            exp_agent[j,:]=xx
        s_=state_next(agent_action)
        r=reward()
        uav_x,uav_y=loacation_back()
        reward_save[episode*max_step+step]=r
        temp_sr=np.zeros((n_agents,N_STATES*2+1))
        for j in range(n_agents):
            temp_sr[j,:]=np.hstack((s,s_,r))
        exp=np.hstack((exp_agent,temp_sr))
        qmix.store_transition(exp)

        while qmix.memory_counter >100:
            print('无人机位置', uav_x, uav_y)  # MEMORY_CAPACITY/20
            print('episode=', episode, 'step=', step, '| reward=: ', r)
            traintime = time.time()
            if episode > 100:
                EPSILON = 0.9 - (episode - 100) * 0.8 / 500
            if EPSILON <= 0.1:
                EPSILON = 0.1
            qmix.learn()
            traintime=time.time()-traintime
            # print('train time= ',traintime)
            break
    Epi_reward[episode]=np.sum(reward_save[np.arange(episode*max_step,episode*max_step+max_step)])/100

data_write_csv("x1.csv", x1_record)
data_write_csv("y1.csv", y1_record)
data_write_csv("x2.csv", x2_record)
data_write_csv("y2.csv", y2_record)
data_write_csv("x3.csv", x3_record)
data_write_csv("y3.csv", y3_record)

np.save('Epi_reward',Epi_reward)


