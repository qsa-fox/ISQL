import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import tqdm
from time import time
import isql_td3 as TD3
from pathlib import Path
import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from gpustat import GPUStatCollection
# gpus = GPUStatCollection.new_query()
# best_gpu = min(gpus, key=lambda x: x.utilization).index
# device = f"cuda:{best_gpu}" if gpus else "cpu"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

envs_locomotion = [
    "hopper-medium-v2",
    "walker2d-medium-v2",
    "halfcheetah-medium-v2",

    "halfcheetah-medium-expert-v2",
    "hopper-medium-expert-v2",    
    "walker2d-medium-expert-v2",

    "halfcheetah-medium-replay-v2",
    "hopper-medium-replay-v2",
    "walker2d-medium-replay-v2",
]

envs_antmaze = [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
]


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.rtg = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
        )
    
    def calculate_rtg(self):
        pass


    def sample_with_rtg(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.rtg[ind]).to(self.device),
        )

    def sample2(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind
        )

    def convert_D4RL(self, dataset, r_scale=1.0, r_shift=0):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.next_action = dataset['next_actions']
        self.reward = dataset['rewards'].reshape(-1, 1) * r_scale + r_shift
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_rtg_bak(self, dataset, r_scale=1.0, r_shift=0, gamma=0.99):
        dataset = list(dataset)
        for seq in dataset:
            len_seq = len(seq['observations'])
            self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
            self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
            self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
            self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
            self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
            self.not_done[self.ptr:self.ptr+len_seq] = 1 - (seq['terminals'] | seq['timeouts']).copy().reshape(-1, 1)
            
            rtg = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
            for i in range(1, len(rtg)):
                rtg[-i-1] = rtg[-i-1] + gamma * rtg[-i]
            self.rtg[self.ptr:self.ptr+len_seq] = rtg

            self.ptr += len_seq

        self.size = self.ptr


    def convert_D4RL_rtg(self, dataset, r_scale=1.0, r_shift=0, gamma=0.99):
        dataset = list(dataset)
        for seq in dataset:
            len_seq = len(seq['observations'])
            self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
            self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
            self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
            self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
            self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
            self.not_done[self.ptr:self.ptr+len_seq] = 1 - (seq['terminals'] | seq['timeouts']).copy().reshape(-1, 1)
            self.not_done[self.ptr+len_seq] = 0
            
            rtg = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
            for i in range(1, len(rtg)):
                rtg[-i-1] = rtg[-i-1] + gamma * rtg[-i]
            self.rtg[self.ptr:self.ptr+len_seq] = rtg

            self.ptr += len_seq

        self.size = self.ptr


    def convert_D4RL_sparse(self, dataset, r_scale=1.0, r_shift=0):
        dataset = list(dataset)
        for seq in dataset:
            if seq['rewards'].sum() == 0:
                len_seq = len(seq['observations'])
                self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
                self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
                self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
                self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
                self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
                self.not_done[self.ptr:self.ptr+len_seq] = 1 - seq['terminals'].copy().reshape(-1, 1)
                self.ptr += len_seq
        self.size_no_reward = self.ptr
        for seq in dataset:
            if seq['rewards'].sum() != 0:
                len_seq = len(seq['observations'])
                self.state[self.ptr:self.ptr+len_seq] = seq['observations'].copy()
                self.action[self.ptr:self.ptr+len_seq] = seq['actions'].copy()
                self.next_state[self.ptr:self.ptr+len_seq] = np.concatenate([seq['observations'][1:], seq['observations'][-1:]], axis=0).copy()
                self.next_action[self.ptr:self.ptr+len_seq] = np.concatenate([seq['actions'][1:], seq['actions'][-1:]], axis=0).copy()
                self.reward[self.ptr:self.ptr+len_seq] = seq['rewards'].copy().reshape(-1, 1) * r_scale + r_shift
                self.not_done[self.ptr:self.ptr+len_seq] = 1 - seq['terminals'].copy().reshape(-1, 1)
                self.ptr += len_seq
        self.size_reward = self.ptr - self.size_no_reward
        self.size = self.ptr

    def sample_sparse(self, batch_size, reward_ratio=0.5):
        ind_no_reward = np.random.randint(0, self.size_no_reward, size=int(batch_size*(1-reward_ratio)))
        ind_reward = np.random.randint(self.size_no_reward, self.size, size=batch_size-len(ind_no_reward))
        ind = np.concatenate([ind_no_reward, ind_reward])

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
        )

    def normalize_states(self, eps=1e-5):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        # Qs = []
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            # Qs.append(policy.critic.Q1(torch.FloatTensor(state).to(device), torch.FloatTensor(action[None]).to(
                # device)).detach().cpu().numpy().squeeze())
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    return d4rl_score


def eval_policy_list(policy, env_name, seed, mean, std, eval_episodes=20):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)

    score_list = {'ddpg': [], 'awr': []}
        
    for actor_type in ['ddpg', 'awr']:
        for i in range(len(policy.actor_list[actor_type])):
            avg_reward = 0.
            for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                # Qs = []
                while not done:
                    state = (np.array(state).reshape(1, -1) - mean) / std
                    action = policy.select_action(state, actor_indice=i, actor_type=actor_type)
                    # Qs.append(policy.critic.Q1(torch.FloatTensor(state).to(device), torch.FloatTensor(action[None]).to(
                        # device)).detach().cpu().numpy().squeeze())
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward

            avg_reward /= eval_episodes
            d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
            score_list[actor_type].append(d4rl_score)

    print("---------------------------------------")
    print(f"episodes: {eval_episodes}, D4RL score_list: {score_list}")
    print("---------------------------------------")
    return score_list


def format_dict_values(d):
    return {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in d.items()}

def main(args):
    file_name = f"{args.env}__seed{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./infos"):
        os.makedirs("./infos")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_freq": args.policy_freq,
        "beta_list": args.beta_list,
        "critic_adv_beta": args.critic_adv_beta,
    }
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    dataset = d4rl.qlearning_dataset(env)

    replay_buffer.convert_D4RL(dataset, r_scale=args.reward_scale, r_shift=args.reward_shift)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    # Initialize policy
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    info_log = []
    t0 = time()
    for t in (range(int(args.max_timesteps))):
        info = policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            info_log.append(info)
            eopch_time = (time() - t0) * 1000 / args.eval_freq
            score_list = eval_policy_list(policy, args.env, args.seed, mean, std)
            for actor_type in ['ddpg', 'awr']:
                for i in range(len(args.beta_list)):
                    info[f'score/{actor_type}/beta_{args.beta_list[i]}'] = score_list[actor_type][i]
            wandb.log(info)
            info['epoch_time'] = eopch_time
            print(f"Time steps: {t + 1}", format_dict_values(info))
            if args.save_model: policy.save(f"./models/{file_name}")
            t0 = time()
        if (t + 1) % int(5e5) == 0:
            torch.save(policy, f"./models/{file_name}_steps{t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3")  # Policy name
    parser.add_argument("--env", default="antmaze-large-play-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # TD3
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.999)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_freq", default=5, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--reward_scale", default=1, type=float)
    parser.add_argument("--reward_shift", default=-1, type=float)
    args = parser.parse_args()

    args.beta_list = [0.1]

    for critic_adv_beta in [0.1]:
        args.critic_adv_beta = critic_adv_beta
        for env in envs_antmaze:
            args.env = env
            for seed in [0]:
                args.seed = seed
                wandb.init(project='isql', reinit=True, config=args, 
                                    settings=wandb.Settings(code_dir='.'), name='',
                                    group='isql-chi', mode='offline', save_code=True)
                print(args)
                main(args)
                wandb.finish()
