import torch
import torch.nn as nn
from torch import optim
import numpy as np
from network import Critic,Actor
from Tracker import Cfg
from Env import Env
import torch.nn.functional as F

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

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 500000  # How many last steps to keep for model training

UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MEMORY_FRACTION = 0.20


# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class Agent():
    def __init__(self,opt,
                 tau=0.05):

        self.cfg = Cfg(INPUT_FEATURE_NUM=3,OUTPUT_FEATURE_NUM=3)
        self.critic=Critic(self.cfg).to(opt.device)
        self.target_critic=Critic(self.cfg).to(opt.device)
        self.actor=Actor(self.cfg).to(opt.device)
        self.target_actor = Actor(self.cfg).to(opt.device)
        self.opt=opt
        self.critic_file=opt.critic_file
        self.actor_file=opt.actor_file
        self.tau=tau
        #self.replay_memory = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.replay_memory = []#deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        n_actions= self.actor.output_size
        self.noise = OUActionNoise(mu=np.zeros(n_actions),sigma=0.3,dt=1/self.opt.Fs)

        if (os.path.isfile(self.critic_file)):
            new_state_dict = torch.load(self.critic_file, map_location=self.opt.device)

            for k, v in new_state_dict.items():
                if (k[7:] == 'module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            # load params
            self.critic.load_state_dict(new_state_dict)
            self.target_critic.load_state_dict(new_state_dict)


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
            self.target_actor.load_state_dict(new_state_dict)

        self.target_critic.eval()
        self.target_actor.eval()


    def store_replay_memory(self, transition):
        self.replay_memory.append(transition)
    def reset_replay_memory(self):
        self.replay_memory = []
    # def action(self,state):
    #     self.actor.eval()
    #     action,_ = self.actor.sample_action(state)
    #     return action
    def choose_action(self, state ,train=False):

        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.opt.device)

        if (train):
            mu, _ = self.actor(state)
            mu_prime = mu + torch.tensor(self.noise(),
                                     dtype=torch.float).to(self.opt.device)#.pow(2)

            #mu_prime = torch.exp(mu_prime)
            # mu_prime = torch.clamp(mu_prime,0,self.actor.max_action)
            mu_prime=torch.tensor(self.actor.max_action) * torch.sigmoid(mu_prime)

            #mu_prime = mu_prime
            self.actor.train()
            return mu_prime.cpu().detach().numpy()
        else:
            mu = self.actor.act(state)
            return mu.cpu().detach().numpy()

    def update(self):
        if len(self.replay_memory) < self.opt.batchSize:
            return 0,0

        #sample_inxs = random.sample(range(len(self.replay_memory)),self.opt.batchSize)
        #batch = [self.replay_memory[inx] for inx in sample_inxs]
        batch = random.sample(self.replay_memory,self.opt.batchSize)
        ### transition = (current_state, action, reward, new_state, done) ###

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

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.train()

        current_Q = self.critic(current_states, action)
        future_actions = self.target_actor.act(next_states)
        future_Q = self.target_critic(next_states, future_actions)

        #calculate Bellman equation
        new_q=reward + (torch.ones_like(done,device=self.opt.device) - done) * DISCOUNT * future_Q

        #update critic
        self.critic_optimizer.zero_grad()
        critic_loss=nn.MSELoss()(new_q,current_Q)
        critic_loss.backward()
        self.critic_optimizer.step()

        # update
        #self.target_critic.train()
        #self.critic.eval()
        # for p in self.critic.parameters():
        #     p.requires_grad = False

        self.actor_optimizer.zero_grad()
        new_action = self.actor.act(current_states)

        actor_loss = self.critic(current_states, new_action)
        actor_loss = -torch.mean(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()

        # for p in self.critic.parameters():
        #     p.requires_grad = True


        ### update slowly with tau parameter ##
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)



        for name in critic_state_dict:
            target_critic_dict[name] = self.tau * critic_state_dict[name].clone() + \
                                      (1 - self.tau) * target_critic_dict[name].clone()


        for name in actor_state_dict:
            target_actor_dict[name] = self.tau * actor_state_dict[name].clone() + \
                                     (1 - self.tau) * target_actor_dict[name].clone()


        return critic_loss.item(), actor_loss.item()

def trainOneAgent(env,agent,rewards=[],critics_losses=[],actor_losses=[],i=0):

        current_state = env.reset()
        critic_total_loss = critics_losses[i]
        actor_total_loss = actor_losses[i]
        episode_reward = rewards[i]
        if (len(agent.replay_memory)>REPLAY_MEMORY_SIZE):
            print('Reseting memory')
            agent.reset_replay_memory()
        agent.noise.reset()
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
        agents.append(Agent(opt))
        critics_losses.append(0)
        actor_losses.append(0)
        rewards.append(0)

    EPISODES = 1000#len(dataloader)


    for episode in range(1,EPISODES+1):
        start=time.time()
        processes=[]

        for i,(env,agent) in enumerate(zip(envs,agents)):
            trainOneAgent(env,agent,rewards,critics_losses,actor_losses,i)
            #p = Thread(target=trainOneAgent, args=(env,agent,critics_loss,actor_loss,reward))
            #p.start()
            #processes.append(p)

        #for p in processes:
        #    p.join()



        print('steps={}'.format(envs[0].current))

        N = len(agents)
        target_actor=agents[0].target_actor
        target_critic=agents[0].target_critic
        target_actor_params = target_actor.named_parameters()
        target_critic_params = target_critic.named_parameters()


        target_critic_dict_avg = dict(target_critic_params)
        target_actor_dict_avg = dict(target_actor_params)

        for agent in agents[1:]:
            target_actor_params = agent.target_actor.named_parameters()
            target_critic_params = agent.target_critic.named_parameters()

            target_critic_dict = dict(target_critic_params)
            target_actor_dict = dict(target_actor_params)

            for name in target_critic_dict_avg:
                target_critic_dict_avg[name] = target_critic_dict_avg[name].clone() \
                                               + 1 / N * target_critic_dict[name].clone()
            for name in target_actor_dict_avg:
                target_actor_dict_avg[name] = target_actor_dict_avg[name].clone() \
                                             +1/N*target_actor_dict[name].clone()

        rewards_ = np.mean(np.array(rewards))
        critics_losses_ = np.mean(np.array(critics_losses))
        actor_losses_ = np.mean(np.array(actor_losses))
        if (best_rewards < rewards_):
            torch.save(target_critic.state_dict(), '%s' % opt.critic_file)
            torch.save(target_actor.state_dict(), '%s' % opt.actor_file)
            best_rewards = rewards_
            np.save('best_rewards.npy', rewards_)
        print ('Episode = {}/{} Episode reward = {} Critic loss avg={} Actor loss avg={} Elap time={}'.format(episode,EPISODES+1,rewards_,critics_losses_,actor_losses_,time.time()-start))

def validation_ActorCritic(opt):
    Visualization = True
    env = Env(opt)
    agent = Agent(opt)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=True, help='Test Mode')
    parser.add_argument('--AI_denoising', type=bool, default=False, help='Denoising based AI')
    parser.add_argument('--use_particle_filter', type=bool, default=False, help='Use particle filter')
    parser.add_argument('--K', type=int, default=16, help='Number for samples for averaging')
    parser.add_argument('--location_file', type=str, default='./expirementV3/SensorArrayMeasurements.csv',
                        help='path to location and oriantaion file if 0 read location TCP')
    parser.add_argument('--expirement_file', type=str, default='./expirementV3/D75R2O3.mat',
                        help='path to expirement measures')
    parser.add_argument('--landmark_file', type=str, default='./expirementV3/landmarks2.csv',
                        help='path to landmark location')

    parser.add_argument('--use_velocity', type=bool, default=True, help='use velocity to state model')
    parser.add_argument('--velocity_record', type=bool, default=False, help='use velocity to state model')

    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--batchSize', type=int, default=256, help='batchSize')

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
    parser.add_argument('--fileName', type=str, default='C:/Users/STL/Desktop/traking/trajectory5.mat', help='trajectory file')
    parser.add_argument('--smoothing', type=bool, default=True, help='use smoothing')

    parser.add_argument('--critic_file', type=str, default='critic.pth', help='critic_file')
    parser.add_argument('--actor_file', type=str, default='actor.pth', help='actor_file')

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.fc_list = [256,266,276,286,296,306]

    opt.acc_xy = 10
    opt.acc_z = 1
    opt.theta_vel = 0.392699081698724
    opt.phi_vel = 0.392699081698724
    opt.sigma_mag = 0.01
    opt.sigma_rho = 1
    opt.MinNumberOfSensor=3
    opt.remove_low_snr_sensor=False
    opt.count_high_diff=[0]


    train_ActorCritic(opt)
    validation_ActorCritic(opt)



