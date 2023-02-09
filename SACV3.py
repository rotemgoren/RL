from collections import deque
from Utils import *
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from network import Critic,Actor
from Tracker import Cfg
from Env import Env
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch.utils.data as data
import os
import random
import time
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from multiprocessing import Process
from threading import Thread

import tqdm
from torch.distributions import Categorical

DISCOUNT=0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training

UPDATE_TARGET_EVERY = 20  # Terminal states (end of episodes)
MEMORY_FRACTION = 0.20


# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

class Memory():
    def __init__(self, opt):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states=[]
        self.dones = []
        self.device = opt.device
        self.batchSize=opt.batchSize

    # def __getitem__(self, index):
    #     states=self.states[index]
    #     actions = self.actions[index]
    #     rewards= self.rewards[index]
    #     next_states = self.next_states[index]
    #     dones = self.dones[index]
    #     return states, actions, rewards, next_states, dones

    def sample(self):
        #batch = random.sample(self.replay_memory,self.opt.batchSize)
        indexes = random.sample(range(len(self.states)), self.batchSize)
        #for index in indexes:
        #    yield self.states[index],self.actions[index],self.rewards[index],self.next_states[index],self.dones[index]
        states = torch.cat([self.states[index].unsqueeze(0) for index in indexes])
        actions = torch.cat([self.actions[index] for index in indexes])
        rewards = torch.cat([self.rewards[index] for index in indexes])
        next_states=torch.cat([self.next_states[index].unsqueeze(0) for index in indexes])
        dones = torch.cat([self.dones[index] for index in indexes])
        return states,actions,rewards,next_states,dones

    def __len__(self):
        return len(self.states)

    def store_memory(self, state, action, reward, next_state, done):

        state = torch.FloatTensor(state).to(device=self.device)
        action = torch.FloatTensor(action).to(device=self.device)
        reward = torch.FloatTensor([reward]).to(device=self.device)
        next_state = torch.FloatTensor(next_state).to(device=self.device)
        done = torch.FloatTensor([done]*1).to(device=self.device)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []



class Agent():
    def __init__(self,opt,action_dim=3,
                 lr = 1e-5,
                 tau=0.05,
                 pretrain=False):
        self.opt = opt
        self.critic_1_file=opt.critic_1_file
        self.critic_2_file = opt.critic_2_file
        self.actor_file=opt.actor_file
        self.cfg = Cfg(INPUT_FEATURE_NUM=opt.INPUT_FEATURE_NUM,OUTPUT_FEATURE_NUM=3,dropout=0.5,
                       device_name=opt.device,SAMPLE_NUM=opt.K)

        self.cfg.pretrain = pretrain
        self.cfg.pretrain_file = './weights/BEST_encoder_unet_16samp_mean_{}_{}dim_in_{}dim_out_0.5dropout.pth'.format(self.cfg.pooling,INPUT_FEATURE_NUM,INPUT_FEATURE_NUM)

        self.critic_1 = Critic(self.cfg).to(opt.device)
        self.critic_2 = Critic(self.cfg).to(opt.device)
        self.target_critic_1 = Critic(self.cfg).to(opt.device)
        self.target_critic_2 = Critic(self.cfg).to(opt.device)

        self.actor=Actor(self.cfg).to(opt.device)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.opt.device)
        self.alpha = self.log_alpha.exp()

        self.tau=tau
        self.target_entropy = np.log(self.actor.max_action)
        self.replay_memory = []#Memory(opt)
        #self.replay_memory = data.DataLoader(self.replay_memory,batch_size=opt.batchSize)

        self.target_update_counter = 0
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        if (os.path.isfile(self.critic_1_file)):
            new_state_dict = torch.load(self.critic_1_file, map_location=self.opt.device)

            for k, v in new_state_dict.items():
                if (k[7:] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            self.critic_1.load_state_dict(new_state_dict)
            self.target_critic_1.load_state_dict(new_state_dict)

        if (os.path.isfile(self.critic_2_file)):
            new_state_dict = torch.load(self.critic_2_file, map_location=self.opt.device)

            for k, v in new_state_dict.items():
                if (k[7:] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            self.critic_2.load_state_dict(new_state_dict)
            self.target_critic_2.load_state_dict(new_state_dict)


        if (os.path.isfile(self.actor_file)):
            new_state_dict = torch.load(self.actor_file, map_location=self.opt.device)

            for k, v in new_state_dict.items():
                if (k[7:] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            self.actor.load_state_dict(new_state_dict)


        self.target_critic_1.eval()
        self.target_critic_2.eval()


    def store_replay_memory(self, transition):
        self.replay_memory.append(transition)
        #self.replay_memory.store_memory(transition[0], transition[1], transition[2], transition[3], transition[4])
    def reset_replay_memory(self):
        self.replay_memory = []
        #self.replay_memory.clear_memory()
    def choose_action(self, state ,train=False):

        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.opt.device)
        if (train):
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
            #mu_prime = mu_prime
            self.actor.train()

        else:
            #mu = mu.pow(2)
            actions = self.actor.act(state)


        return actions.detach().cpu().numpy()

    def update(self):
        if len(self.replay_memory) < self.opt.batchSize:
            return 0,0


        ### transition = (current_state, action, reward, new_state, done) ###
        batch = random.sample(self.replay_memory,self.opt.batchSize)
        current_states = np.array([transition[0] for transition in batch])
        action = np.array([transition[1] for transition in batch])
        reward = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        done = np.array([transition[4]*1 for transition in batch])

        current_states = torch.FloatTensor(current_states).to(self.opt.device)
        action = torch.FloatTensor(action[:,0,:]).to(self.opt.device)
        reward = torch.FloatTensor(reward).to(self.opt.device)
        next_states = torch.FloatTensor(next_states).to(self.opt.device)
        done = torch.FloatTensor(done).to(self.opt.device)

        #current_states,action,reward,next_states,done = self.replay_memory.sample()


        # update critics
        self.critic_1.train()
        self.critic_2.train()


        current_q1 = self.critic_1.forward(current_states, action)
        current_q2 = self.critic_2.forward(current_states, action)

        future_actions, log_probs = self.actor.sample_normal(next_states, reparameterize=False)

        future_q1 = self.target_critic_1(next_states, future_actions)
        future_q2 = self.target_critic_2(next_states, future_actions)

        future_Q = torch.min(future_q1,future_q2)
        #calculate Bellman equation
        q_target=reward + (torch.ones_like(done,device=self.opt.device) - done) * DISCOUNT * (future_Q - self.alpha *log_probs)


        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_1_loss = 0.5 * nn.MSELoss()(current_q1, q_target)
        critic_2_loss = 0.5 * nn.MSELoss()(current_q2, q_target)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward(retain_graph=True)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # for p1,p2 in zip(self.critic_1.parameters(),self.critic_2.parameters()):
        #     p1.requires_grad = False
        #     p2.requires_grad = False

        new_actions, new_log_probs = self.actor.sample_normal(current_states, reparameterize=True)
        q1_new_policy = self.critic_1.forward(current_states, new_actions)
        q2_new_policy = self.critic_2.forward(current_states, new_actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = self.alpha *new_log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # for p1,p2 in zip(self.critic_1.parameters(),self.critic_2.parameters()):
        #     p1.requires_grad = True
        #     p2.requires_grad = True

        self.temp_optimizer.zero_grad()
        temp_loss = -self.log_alpha * (new_log_probs.detach() + self.target_entropy).mean()
        temp_loss.backward()
        self.temp_optimizer.step()
        self.alpha = self.log_alpha.exp()


        ### update slowly with tau parameter ##
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_critic_1_dict = dict(target_critic_1_params)
        target_critic_2_dict = dict(target_critic_2_params)



        for name in critic_1_state_dict:
            target_critic_1_dict[name] = self.tau * critic_1_state_dict[name].clone() + \
                                      (1 - self.tau) * target_critic_1_dict[name].clone()


        for name in critic_2_state_dict:
            target_critic_2_dict[name] = self.tau * critic_2_state_dict[name].clone() + \
                                     (1 - self.tau) * target_critic_2_dict[name].clone()


        return critic_loss.item(), actor_loss.item()

def trainOneAgent(env,agent,rewards=[],critics_losses=[],actor_losses=[],i=0):

        current_state = env.reset()
        critic_total_loss = critics_losses[i]
        actor_total_loss = actor_losses[i]
        episode_reward = rewards[i]
        if (len(agent.replay_memory)>REPLAY_MEMORY_SIZE):
            print('Reseting memory')
            agent.reset_replay_memory()
        done = False
        agent.actor.eval()
        steps=0
        while not done: #while not env.current == env.totalSamples:
            action=agent.choose_action(current_state,train=True)

            #Qs,Qv = env.tracker.getStateCov()
            #action = np.array([np.concatenate((np.diag(Qs),np.diag(Qv)))])

            new_state, reward, done = env.step(action)

            episode_reward += reward
            agent.store_replay_memory((current_state, action, reward, new_state, done))
            if steps % UPDATE_TARGET_EVERY==0:
                critic_loss,actor_loss=agent.update()
                critic_total_loss += critic_loss
                actor_total_loss += actor_loss
            steps+=1
            current_state = new_state

        rewards[i] = episode_reward / env.current
        critics_losses[i] = critic_total_loss / env.current
        actor_losses[i] = actor_total_loss / env.current



def train_ActorCritic(opt):

    if os.path.isfile('best_rewards.npy'):
        best_rewards = np.load('best_rewards.npy')

    else:
        best_rewards = -np.inf
    print("best_rewards={}".format(best_rewards))
    envs=[]
    agents=[]
    critics_losses=[]
    actor_losses=[]
    rewards=[]
    for _ in range(1):
        envs.append(Env(opt))
        agents.append(Agent(opt,pretrain=True))
        critics_losses.append(0)
        actor_losses.append(0)
        rewards.append(0)


    for episode in range(1,opt.EPISODES+1):
        start=time.time()
        processes=[]

        for i,(env,agent) in enumerate(zip(envs,agents)):
            trainOneAgent(env,agent,rewards,critics_losses,actor_losses,i)
            #p = Thread(target=trainOneAgent, args=(env,agent,critics_loss,actor_loss,reward))
            #p.start()
            #processes.append(p)

        #for p in processes:
        #    p.join()



        print('alpha = {} steps={}'.format(agents[0].alpha.item(),envs[0].current))
        N = len(agents)
        actor = agents[0].actor
        target_critic_1=agents[0].target_critic_1
        target_critic_2=agents[0].target_critic_2

        actor_params = actor.named_parameters()
        target_critic_1_params = target_critic_1.named_parameters()
        target_critic_2_params = target_critic_2.named_parameters()

        actor_dict_avg = dict(actor_params)
        target_critic_1_dict_avg = dict(target_critic_1_params)
        target_critic_2_dict_avg = dict(target_critic_2_params)

        for agent in agents[1:]:
            actor_params = agent.actor.named_parameters()
            target_critic_1_params = agent.target_critic_1.named_parameters()
            target_critic_2_params = agent.target_critic_2.named_parameters()

            actor_dict = dict(actor_params)
            target_critic_1_dict = dict(target_critic_1_params)
            target_critic_2_dict = dict(target_critic_2_params)

            for name in actor_dict_avg:
                actor_dict_avg[name] = actor_dict_avg[name].clone() \
                                               + 1 / N * actor_dict[name].clone()

            for name in target_critic_1_dict_avg:
                target_critic_1_dict_avg[name] = target_critic_1_dict_avg[name].clone() \
                                               + 1 / N * target_critic_1_dict[name].clone()
            for name in target_critic_2_dict_avg:
                target_critic_2_dict_avg[name] = target_critic_2_dict_avg[name].clone() \
                                             +1/N*target_critic_2_dict[name].clone()


        rewards_ = np.mean(np.array(rewards))
        critics_losses_ = np.mean(np.array(critics_losses))
        actor_losses_ = np.mean(np.array(actor_losses))
        #if (best_rewards < rewards_):
        torch.save(target_critic_1.state_dict(), '%s' % opt.critic_1_file)
        torch.save(target_critic_2.state_dict(), '%s' % opt.critic_2_file)
        torch.save(actor.state_dict(), '%s' % opt.actor_file)
        best_rewards = rewards_
        np.save('best_rewards.npy',rewards_)
        #print("BEST MODEL SAVED")
        print ('Episode = {}/{} Episode reward = {} Critic loss avg={} Actor loss avg={} Elap time={}'.format(episode,opt.EPISODES+1,rewards_,critics_losses_,actor_losses_,time.time()-start))

def validation_ActorCritic(opt):
    if os.path.isfile('best_rewards.npy'):
        best_rewards = np.load('best_rewards.npy')

    else:
        best_rewards = -np.inf
    print("best_rewards={}".format(best_rewards))
    env = Env(opt)
    agent = Agent(opt)

    Visualization = True
    start=time.time()
    current_state = env.reset()
    if (len(agent.replay_memory)>REPLAY_MEMORY_SIZE):
        print('Reseting memory')
        agent.reset_replay_memory()
    done = False
    agent.actor.eval()
    total_reward = 0
    action = agent.choose_action(current_state)
    new_state, reward, done = env.step(action)
    xx= new_state.copy()
    current_state = new_state
    while not done:
        action=agent.choose_action(current_state)

        new_state, reward, done = env.step(action)
        xx = np.vstack((xx,new_state[-1,:]))
        total_reward += reward

        current_state = new_state

    print ('Episode reward = {} step={} Elap time={}'.format(total_reward/env.current,env.current,time.time()-start))
    xx=xx.T + np.array([[(env.maxX-env.minX)/2,(env.maxY-env.minY)/2,(env.maxZ-env.minZ)/2]]).T
    print(np.std(xx[:, :] - env.trajectory[:3, 1:-1],axis=1).mean())
    if (Visualization):
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        ax.scatter(env.trajectory[0, 1:], env.trajectory[1, 1:], env.trajectory[2, 1:], c='r', marker='o', s=3)
        ax.scatter(xx[0, 1:], xx[1, 1:], xx[2, 1:], c='b', marker='o', s=3)
        plt.title('3D quiver')
        plt.show()



K = 16
INPUT_FEATURE_NUM=3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=True, help='Test Mode')
    parser.add_argument('--AI_denoising', type=bool, default=False, help='Denoising based AI')
    parser.add_argument('--use_particle_filter', type=bool, default=False, help='Use particle filter')

    parser.add_argument('--location_file', type=str, default='./expirementV3/SensorArrayMeasurements.csv',
                        help='path to location and oriantaion file if 0 read location TCP')
    parser.add_argument('--expirement_file', type=str, default='./expirementV3/D75R2O3.mat',
                        help='path to expirement measures')
    parser.add_argument('--landmark_file', type=str, default='./expirementV3/landmarks2.csv',
                        help='path to landmark location')
    parser.add_argument('--use_velocity', type=bool, default=True, help='use velocity to state model')
    parser.add_argument('--velocity_record', type=bool, default=False, help='use velocity to state model')

    parser.add_argument('--smoothing', type=bool, default=True, help='use smoothing')

    parser.add_argument('--device', type=str, default='cuda:1', help='gpu device')
    parser.add_argument('--batchSize', type=int, default=128, help='batchSize')
    parser.add_argument('--use_dynamic_AI_state_cov', type=bool, default=True, help='use dynamic AI state cov')

    parser.add_argument('--Mmag', type=float, default=(8, 20), help='Range of Mmag')
    parser.add_argument('--nInst', type=int, default=15, help='Number of instances for dipole location')
    parser.add_argument('--BW', type=float, default=2.5, help='Band Width')
    parser.add_argument('--sensor_noise', type=float, default=50, help='sensor noise ASD [fT/Hz]')
    parser.add_argument('--Fs', type=float, default=5, help='Sampling Freq [Hz]')
    parser.add_argument('--nSens', type=int, default=4, help='Number of Sensors')
    parser.add_argument('--use_torch', type=bool, default=False, help='Use Pytorch')
    parser.add_argument('--northEastUp2NorthEastDown', type=bool, default=True, help='Use NorthEastUp to NorthEastDown')
    parser.add_argument('--deg2Rad', type=bool, default=True, help='Use Degree to Radian ')
    parser.add_argument('--complex', type=bool, default=True, help='Use Complex Data')
    parser.add_argument('--senderTimeStamp', type=bool, default=False, help='User Timestamp send')
    parser.add_argument('--num_drop_samples', type=int, default=15, help='Number of samples to drop')
    parser.add_argument('--Model', type=int, default=0,
                        help='Magnetic Field Model 0-Free space 1-Full conductive space 2-Half conductive space')
    parser.add_argument('--fc', type=float, default=254.0, help='carrier frequncy [Hz]')
    parser.add_argument('--rho', type=float, default=(10, 100000), help='carrier frequncy [Hz]')
    parser.add_argument('--fileName', type=str, default='C:/Users/STL/Desktop/traking/trajectory6.mat', help='trajectory file')

    parser.add_argument('--critic_1_file', type=str, default='./weights/critic_1_{}_{}dim_in.pth'.format(K,INPUT_FEATURE_NUM), help='critic_1_file')
    parser.add_argument('--critic_2_file', type=str, default='./weights/critic_2_{}_{}dim_in.pth'.format(K,INPUT_FEATURE_NUM), help='critic_2_file')
    parser.add_argument('--actor_file', type=str, default='./weights/actor_{}_{}dim_in.pth'.format(K,INPUT_FEATURE_NUM), help='actor_file')
    parser.add_argument('--value_file', type=str, default='./weights/value.pth', help='value_file')
    parser.add_argument('--EPISODES', type=int, default=1000, help='number of episodes')


    opt = parser.parse_args()
    opt.fc_list = [256,266,276,286,296,306]
    opt.K=K
    opt.INPUT_FEATURE_NUM=INPUT_FEATURE_NUM

    opt.acc_xy = 10
    opt.acc_z = 1
    opt.theta_vel = 0.392699081698724
    opt.phi_vel = 0.392699081698724
    opt.sigma_mag = 0.01
    opt.sigma_rho = 1
    opt.MinNumberOfSensor=3
    opt.remove_low_snr_sensor=False
    opt.count_high_diff=[0]

    writer = SummaryWriter('runs/SACV3_{}_{}dim'.format(opt.K,opt.INPUT_FEATURE_NUM))

    train_ActorCritic(opt)
    validation_ActorCritic(opt)

    writer.close()
