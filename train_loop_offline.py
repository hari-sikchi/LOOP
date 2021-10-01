import numpy as np
import torch
import gym
import argparse
import os
from policies import get_policy
import sac
import plas_utils
import d4rl
import yaml
from logging_utils.logx import EpochLogger


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def reward_to_return(reward_arr, discount):
    assert type(reward_arr) == np.ndarray
    discount_factor = discount ** np.arange(len(reward_arr))
    return np.sum(reward_arr * discount_factor)

def eval_policy_actor(policy, env_name, seed, eval_episodes=5):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    if hasattr(eval_env, '_max_episode_steps'):
        max_step = eval_env._max_episode_steps
    else:
        max_step = 1000
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_steps=0
        while not done:
            episode_steps+=1
            action = policy.get_action(np.array(state),deterministic=True)
            state, reward, done, _ = eval_env.step(action)
            if(episode_steps>=max_step):
                done=True
            avg_reward += reward

    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(
        "Actor| Evaluation over {} episodes: {}".format(
            eval_episodes,
            avg_reward))
    print("---------------------------------------")
    return avg_reward

def eval_policy(policy, env_name, seed, eval_episodes=5,logger=None):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    if hasattr(eval_env, '_max_episode_steps'):
        max_step = eval_env._max_episode_steps
    else:
        max_step = 1000
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        episode_rew = 0
        i=0
        policy.reset()
        while not done:
            i+=1
            action,_ = policy.get_action(np.array(state))
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            avg_reward += reward
            episode_rew+= reward
            if(i>=max_step):
                done=True
            if 'cost' in info:
                avg_cost += info['cost']
        if logger is not None:
           logger.store(MPCEvaluation = 100*eval_env.get_normalized_score(episode_rew))
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}, Normalized: {}".format(eval_episodes, avg_reward,100*eval_env.get_normalized_score(avg_reward)))
    print("---------------------------------------")
    return avg_reward, avg_cost



def run_loop(args):

    config = load_config(args.config)
    exp_name = 'results/offline_loop/'+args.env+'/PETS_dynamics'
    logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':exp_name}
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(
        args.policy, args.env, args.seed))
    print("---------------------------------------")

    # Set seeds
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Create replay buffer
    replay_buffer = sac.ReplayBuffer(state_dim, action_dim,int(2e6))

    # Offline LOOP - fill replay buffer
    dataset = d4rl.qlearning_dataset(env)
    print("Dataset size: {}".format(dataset['observations'].shape[0]))
    replay_buffer.state[:dataset['observations'].shape[0],:] = dataset['observations']
    replay_buffer.action[:dataset['actions'].shape[0],:] = dataset['actions']
    replay_buffer.next_state[:dataset['next_observations'].shape[0],:] = dataset['next_observations']
    replay_buffer.reward[:dataset['rewards'].shape[0]] = dataset['rewards']
    replay_buffer.done[:dataset['terminals'].shape[0]] = dataset['terminals']
    replay_buffer.size =dataset['observations'].shape[0]
    replay_buffer.ptr = (replay_buffer.size+1)%(replay_buffer.max_size)


    # Choose a controller
    policy, offline_policy, dynamics, _ = get_policy(args,  env, replay_buffer, config, policy_name=args.policy)
    
    policy.prior_type = args.prior_type
    policy.offline_policy_type = args.offline_algo



    # Train dynamics model if not already present
    if os.path.exists('results/offline_loop/'+args.env+"/CRR/pyt_save/dynamics.pt"):
        print("Loading saved dynamics model")
        dynamics = torch.load('results/offline_loop/'+args.env+"/CRR/pyt_save/dynamics.pt",map_location=device)
    else:
        logger.setup_pytorch_multiple_saver([dynamics],['dynamics'])
        _, _ = dynamics.train_low_mem()
        logger.save_state({'env': env}, None)
        


    env_name = args.env.split('-')
    # Load the trained value function and policy for offline RL 
    if args.offline_algo=='CRR':
        if len(env_name)==4:
            offline_policy.ac = torch.load('offline_models/crr/' + env_name[0]+'-'+env_name[1]+'-'+env_name[2]+'/crr/corr2_my_policy_beta_2_s'+str(args.seed)+'/pyt_save/model.pt',map_location=device).to(device)    
        else:
            offline_policy.ac = torch.load('offline_models/crr/' + env_name[0]+'-'+env_name[1]+'/crr/corr2_my_policy_beta_2_s'+str(args.seed)+'/pyt_save/model.pt',map_location=device).to(device)    
    elif args.offline_algo=='PLAS':
        if os.path.exists('PLAS_data/PLAS_eval_package/models/vae_v6/'+args.env+"-0_vae.pth"):
            offline_policy.ac = plas_utils.Latent(args.env,state_dim,action_dim)
            offline_policy.ac.load('PLAS_data/PLAS_eval_package',seed=args.seed)
        else:
            print("Offline Q function not available")        


    # Hyperparam search
    horizons = [2,4,10] 
    kappas = [0.01,0.03,0.1,1,3,10]
    sigmas = [0.01,0.05,0.1,0.2,0.4,0.8]
    betas = [0,0.2]
    beta_pessimisms = [0,0.5,1,5]
    for horizon in horizons:
        for kappa in kappas:
            for sigma in sigmas:
                for beta in betas:
                    for beta_pessimism in beta_pessimisms:
                        policy.reinitialize(horizon,kappa,sigma,beta,beta_pessimism)
                        eval_policy(policy,args.env, args.seed+np.random.randint(0,5),logger=logger,eval_episodes=10)
                        offline_policy.test_agent(logger=logger)
                        logger.log_tabular('Horizon', horizon)
                        logger.log_tabular('Kappa', kappa)
                        logger.log_tabular('Sigma', sigma)
                        logger.log_tabular('Beta', beta)
                        logger.log_tabular('BetaPessimism', beta_pessimism)
                        logger.log_tabular('MPCEvaluation', with_min_and_max=True)
                        logger.log_tabular('ActorEvaluation', with_min_and_max=True)
                        logger.dump_tabular()





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="LOOP_OFFLINE_ARC")
    parser.add_argument("--env", default="hopper-medium-expert-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--exp_name", default="dump")
    parser.add_argument("--offline_algo", default="CRR")
    parser.add_argument("--prior_type", default="CRR")
    parser.add_argument('--config', '-c', type=str, default='configs/offline_config.yml', help="specify the path to the configuation file of the models")
    args = parser.parse_args()
    run_loop(args)