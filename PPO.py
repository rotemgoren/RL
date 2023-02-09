#from utils_net import *
from Utils import *
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from network import ValueNetwork,Actor
from Tracker import Cfg
from Env import Env
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import random
import time
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from multiprocessing import Process
import torch.utils.data as data
from threading import Thread

import tqdm
from torch.distributions import Categorical

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training

UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 200

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

Visualization=False

class Memory():
    def __init__(self, opt):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = opt.batchSize
        self.device = opt.device

    def sample(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # states = torch.FloatTensor(self.states).to(device=self.device)
        # actions = torch.FloatTensor(self.actions).to(device=self.device)
        # probs = torch.FloatTensor(self.probs).to(device=self.device)
        # vals = torch.FloatTensor(self.vals).to(device=self.device)
        # rewards = torch.FloatTensor(self.rewards).to(device=self.device)
        # dones = torch.FloatTensor(self.dones).to(device=self.device)

        return self.states,self.actions,self.probs,self.vals,self.rewards,self.dones,batches

    def __len__(self):
        return len(self.batches)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Agent():
    def __init__(self,opt,
                 policy_lr = 3e-4, value_lr = 1e-2,policy_clip=0.2,gamma=0.99,gae_lambda=0.95,
                 tau=0.05,target_kl_div=0.01):

        self.cfg = Cfg(INPUT_FEATURE_NUM=opt.INPUT_FEATURE_NUM, OUTPUT_FEATURE_NUM=3, dropout=0.5,
                       device_name=opt.device, SAMPLE_NUM=opt.K)

        self.cfg.pretrain = True
        self.cfg.pretrain_file = './weights/BEST_encoder_unet_16samp_mean_attention_3dim_in_3dim_out_0.5dropout.pth'

        self.critic=ValueNetwork(self.cfg).to(opt.device)

        self.policy_clip=policy_clip
        self.actor=Actor(self.cfg).to(opt.device)

        self.opt=opt
        self.critic_file=opt.critic_file
        self.actor_file=opt.actor_file
        self.gamma=gamma
        self.gae_lambda=gae_lambda
        #self.replay_memory = ReplayBuffer(REPLAY_MEMORY_SIZE)
        self.memory = Memory(opt)
        self.target_update_counter = 0
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)
        self.n_epochs = opt.n_epochs
        self.target_kl_div = target_kl_div

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



    def choose_action(self, state ,train=False):

        self.actor.eval()
        self.critic.eval()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.opt.device)
        if (train):
            action, log_prob = self.actor.sample_normal(state, reparameterize=False)
            value = self.critic(state)
            #mu_prime = mu_prime
            self.actor.train()
            self.critic.train()
            action=action.cpu().detach().numpy()
            return action,log_prob,value


        else:
            #mu = mu.pow(2)
            action = self.actor.act(state)
            action = action.cpu().detach().numpy()
            return action


    def update(self,new_state):

        # sample from old policy and compute gae
        states, actions, old_probs, values, rewards, dones, batches = self.memory.sample()

        next_value=self.critic(torch.FloatTensor(new_state).to(self.opt.device).unsqueeze(0))
        values.append(next_value)
        gae = 0
        advantage = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1-dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[step]) * gae
            advantage.insert(0, gae)

        advantage = torch.cat(advantage).detach()
        values = torch.cat(values).detach()
        states = torch.FloatTensor(np.array(states)).to(self.opt.device).detach()
        old_probs = torch.FloatTensor(old_probs).to(self.opt.device).detach()
        #actions = torch.FloatTensor(actions).to(self.opt.device)

        # update policy and value

        #for batch in batches:
            #actor_loss = self.train_policy(states[batch],old_probs[batch],advantage[batch])
            #critic_loss = self.train_value(states[batch],advantage[batch],values[batch])

        for _ in range(self.n_epochs):
            for batch in batches:
                new_action, new_log_prob = self.actor.sample_normal(states[batch], reparameterize=True)
                critic_value = self.critic(states[batch])

                prob_ratio = (new_log_prob - old_probs[batch]).exp()
                weighted_probs = prob_ratio*advantage[batch]
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = nn.MSELoss()(returns ,critic_value)
                entropy = -new_log_prob.mean()

                total_loss =actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()


        self.memory.clear_memory()
        return actor_loss.item(), critic_loss.item()

    # def train_policy(self,states,old_probs,advantage):
    #     for it in range(self.n_epochs):
    #         new_action, new_log_prob = self.actor.sample_normal(states, reparameterize=True)
    #         new_log_prob=new_log_prob[:,0]
    #         # prob_ratio = new_log_prob.exp() / old_probs[batch].exp()
    #         prob_ratio = (new_log_prob - old_probs).exp()
    #         weighted_probs = advantage * prob_ratio
    #         weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage
    #         actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
    #         self.actor_optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor_optimizer.step()
    #         KL_div = ((new_log_prob - old_probs)*new_log_prob.exp()).mean()
    #         if KL_div >= self.target_kl_div:
    #             break
    #     return actor_loss
    #
    # def train_value(self,states,advantage,values):
    #     for _ in range(self.n_epochs):
    #         critic_value = self.critic(states)
    #         returns = advantage + values
    #         critic_loss = nn.MSELoss()(returns, critic_value)
    #         self.critic_optimizer.zero_grad()
    #         critic_loss.backward()
    #         self.critic_optimizer.step()
    #     return critic_loss


class ParticleSwarmAgents():
    def __init__(self,agents,opt):
        self.w=0.99
        self.c1=2.00
        self.c2=2.00
        self.best_results=1000*np.ones(opt.N_agents)
        self.best_localAgent = [i for i in range(opt.N_agents)]
        self.global_result = 1000
        self.global_agent=0
        self.particle_velocity = [agent.actor.state_dict() for agent in agents]
        self.device = opt.device
    def particleSwarmStep(self,agents,rewards):

        for i,(agent,reward) in enumerate(zip(agents,rewards)):
            if self.best_results[i] > -reward:
                self.best_results[i] = -reward
                self.best_localAgent[i]=agent.actor.state_dict()

            if (self.global_result > self.best_results[i]):
                self.global_result=self.best_results[i]
                self.global_agent=agent.actor.state_dict()


            params_dict = dict(agent.actor.named_parameters())
            for name in params_dict:
                self.particle_velocity[i][name] = self.w*self.particle_velocity[i][name] \
                                            + self.c1 * torch.rand(agent.actor.state_dict()[name].shape,device=self.device) * (self.best_localAgent[i][name] - agent.actor.state_dict()[name]) \
                                            + self.c2 * torch.rand(agent.actor.state_dict()[name].shape,device=self.device) * (self.global_agent[name] - agent.actor.state_dict()[name])

                agent.actor.state_dict()[name] = agent.actor.state_dict()[name]+self.particle_velocity[i][name]

        return agents,self.global_agent

def trainOneAgent(env,agent,rewards=[],critics_losses=[],actor_losses=[],i=0):


        current_state = env.reset()
        critic_total_loss = critics_losses[i]
        actor_total_loss = actor_losses[i]
        episode_reward = rewards[i]
        agent.memory.clear_memory()
        done = False
        n_steps=0
        while not done: #while not env.current == env.totalSamples:
            action,log_prob,value=agent.choose_action(current_state,train=True)
            new_state, reward, done = env.step(action)
            n_steps+=1

            episode_reward += reward
            agent.memory.store_memory(current_state, action, log_prob, value,reward, done)
            if n_steps % N_rollout == 0 :
                actor_loss,critic_loss=agent.update(new_state)
                critic_total_loss += critic_loss
                actor_total_loss += actor_loss

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
    for _ in range(opt.N_agents):
        envs.append(Env(opt))
        agents.append(Agent(opt,policy_lr = 3e-4, value_lr = 1e-2))
        critics_losses.append(0)
        actor_losses.append(0)
        rewards.append(0)

    #psa = ParticleSwarmAgents(agents,opt)



    for episode in range(1,opt.EPISODES+1):
        start=time.time()
        processes=[]

        for i,(env,agent) in enumerate(zip(envs,agents)):
            trainOneAgent(env,agent,rewards,critics_losses,actor_losses,i)
#             p = Process(target=trainOneAgent, args=(env,agent,rewards,critics_losses,actor_losses,i))
#             p.start()
#             processes.append(p)
#
#         for p in processes:
#            p.join()



        #agents,optimal_actor=psa.particleSwarmStep(agents,rewards)

        # N = len(agents)
        # target_actor=agents[0].actor
        # target_critic=agents[0].critic
        #
        #
        # target_actor_params = target_actor.named_parameters()
        # target_critic_params = target_critic.named_parameters()
        #
        #
        # target_critic_dict_avg = dict(target_critic_params)
        # target_actor_dict_avg = dict(target_actor_params)
        #
        #
        # for agent in agents[1:]:
        #     target_actor_params = agent.target_actor.named_parameters()
        #     target_critic_params = agent.target_critic.named_parameters()
        #
        #     target_critic_dict = dict(target_critic_params)
        #     target_actor_dict = dict(target_actor_params)
        #
        #     for name in target_critic_dict_avg:
        #         target_critic_dict_avg[name] = target_critic_dict_avg[name].clone() \
        #                                        + 1 / N * target_critic_dict[name].clone()
        #
        #     for name in target_actor_dict_avg:
        #         target_actor_dict_avg[name] = target_actor_dict_avg[name].clone() \
        #                                      +1/N*target_actor_dict[name].clone()


        print('steps={}'.format(envs[0].current))
        rewards_ = np.mean(np.array(rewards))
        critics_losses_ = np.mean(np.array(critics_losses))
        actor_losses_ = np.mean(np.array(actor_losses))
        writer.add_scalar('Loss/critic',critics_losses_,episode)
        writer.add_scalar('Loss/actor', actor_losses_, episode)
        writer.add_scalar('Loss/rewards', rewards_, episode)

        if (best_rewards < rewards_):
            torch.save(agents[0].critic.state_dict(), '%s' % opt.critic_file)
            torch.save(agents[0].actor.state_dict(), '%s' % opt.actor_file)
            best_rewards = rewards_
            np.save('best_rewards.npy',rewards_)
            print("BEST MODEL SAVED")
        print ('Episode = {}/{} Episode reward = {} Critic loss avg={} Actor loss avg={} Elap time={}'.format(episode,EPISODES+1,rewards_,critics_losses_,actor_losses_,time.time()-start))

def validation_ActorCritic(opt):
    env = Env(opt)
    agent = Agent(opt)


    start=time.time()
    current_state = env.reset()
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
    if (Visualization):
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        ax.scatter(env.trajectory[0, :], env.trajectory[1, :], env.trajectory[2, :], c='r', marker='o', s=3)
        ax.scatter(xx[0, :], xx[1, :], xx[2, :], c='b', marker='o', s=3)
        plt.title('3D quiver')
        plt.show()



K=16
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

    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
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
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')

    parser.add_argument('--Model', type=int, default=0,
                        help='Magnetic Field Model 0-Free space 1-Full conductive space 2-Half conductive space')
    parser.add_argument('--fc', type=float, default=254.0, help='carrier frequncy [Hz]')
    parser.add_argument('--rho', type=float, default=(10, 100000), help='carrier frequncy [Hz]')
    parser.add_argument('--fileName', type=str, default='C:/Users/STL/Desktop/traking/trajectory5.mat', help='trajectory file')

    parser.add_argument('--critic_file', type=str, default='./weights/value_{}.pth'.format(K), help='critic_2_file')
    parser.add_argument('--actor_file', type=str, default='./weights/actor_{}.pth'.format(K), help='actor_file')
    parser.add_argument('--N_agents', type=int, default=1, help='N_agents')
    parser.add_argument('--EPISODES', type=int, default=1000, help='number of episodes')

    opt = parser.parse_args()
    opt.fc_list = [256,266,276,286,296,306]
    opt.K=K
    opt.INPUT_FEATURE_NUM=3

    opt.acc_xy = 10
    opt.acc_z = 1
    opt.theta_vel = 0.392699081698724
    opt.phi_vel = 0.392699081698724
    opt.sigma_mag = 0.01
    opt.sigma_rho = 1
    opt.MinNumberOfSensor=3
    opt.remove_low_snr_sensor=False
    opt.count_high_diff=[0]

    N_rollout = 1 * opt.batchSize
    print("Number of rollouts = {}".format(N_rollout))
    writer = SummaryWriter('runs/ppo_{}'.format(opt.K))

    train_ActorCritic(opt)
    validation_ActorCritic(opt)
    writer.close()