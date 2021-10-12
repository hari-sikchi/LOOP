# Implements LOOP: ARC with H-step lookahead policies for Online RL
import numpy as np
import torch
import gym
import argparse
import os
import time
import sac
import yaml
import envs
from policies import get_policy
from logging_utils.logx import EpochLogger


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

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
    if torch.is_tensor(policy.mean):
        old_mean = policy.mean.clone()
    else:
        old_mean = policy.mean.copy()
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        i=0
        total_score = 0.
        success_score = 0
        policy.reset()
        while not done:
            i+=1
            if(i>=max_step):
                break
            action = policy.get_action(np.array(state),deterministic=True)
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            avg_reward += reward
            if 'goal_achieved' in info:
                logger.store(TestSuccessRate=info['goal_achieved'])
                if info['goal_achieved']:
                    success_score+=1
            if 'pddm' in env_name:
                total_score+=info['score']
            if 'cost' in info:
                avg_cost += info['cost']
        if 'goal_achieved' in info:
            logger.store(TestSuccessPercentage=(success_score>20)*100)
        if 'pddm' in env_name:
            logger.store(Score=total_score)

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    policy.mean = old_mean
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_cost


def run_loop(args):
    start_time = time.time()
    config = load_config(args.config)
    logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(
        args.policy, args.env, args.seed))
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = sac.ReplayBuffer(state_dim, action_dim,int(1e6))

    # Choose a controller
    policy, sac_policy, dynamics, lookahead_policies = get_policy(args,  env, replay_buffer, config, policy_name=args.policy)
    

    # Noise to be added to controller while executing trajectory
    noise_amount = config['mpc_config']['exploration_noise']

    total_timesteps = 0
    episode_timesteps = 0
    episode_reward, episode_cost = 0, 0
    evaluation_rewards, evaluation_costs = 0, 0
    evaluation_episodes = 0
    success_score=0
    state, done, done_episode = env.reset(), False, False

    for t in range(int(args.max_timesteps)):
        total_timesteps += 1
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.get_action(np.array(state))
            action = action + np.random.normal(action.shape) * noise_amount
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high)

        # Take the safe action
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        if 'cost' in info:
            episode_cost += info['cost']

        if hasattr(env, '_max_episode_steps'):
            if 'goal_achieved' in info:
                logger.store(SuccessRate=info['goal_achieved'])
            done_bool = float(
                done) if episode_timesteps < env._max_episode_steps else 0

            if episode_timesteps >= env._max_episode_steps or done:
                done_episode=True
        else:
            done_bool = float(
                done) if episode_timesteps < 1000 else 0

            if episode_timesteps >= 1000 or done:
                done_episode=True

        # Store data in replay buffer
        replay_buffer.store(state, action, reward,next_state,  done_bool, cost=info.get('cost',0))
        state = next_state

        if (t+1) % args.dynamics_freq == 0:
            # dynamics_trainloss,dynamics_valloss = dynamics.train_low_mem() # Low memory alternative
            dynamics_trainloss,dynamics_valloss = dynamics.train()
            logger.store(DynamicsTrainLoss = dynamics_trainloss, DynamicsValLoss = dynamics_valloss)


        if args.policy in lookahead_policies:
            if t >= args.start_timesteps and t%sac_policy.update_every==0:
                sac_policy.train()

        if done_episode:
            policy.reset()
            evaluation_costs += episode_cost
            evaluation_rewards += episode_reward
            logger.store(MPCEvaluation=evaluation_rewards, MPCCostEvaluation=evaluation_costs)
            if 'goal_achieved' in info:
                logger.store(SuccessPercentage=(success_score>20)*100)
            episode_reward, episode_cost = 0, 0
            evaluation_rewards, evaluation_costs = 0,0
            success_score=0
            evaluation_episodes += 1
            state, done = env.reset(), False
            done_episode=False
            episode_timesteps = 0


        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            logger.save_state({'env': env}, None)
            if args.policy in lookahead_policies:
                if(config['sac_config']['evaluation_mode']=='actor') :
                    actor_rew = eval_policy_actor(sac_policy, args.env, args.seed+np.random.randint(0,5))
                    logger.store(ActorEvaluation=actor_rew)
            
            if noise_amount!=0 or 'ARC' in args.policy:
                test_mpc_eval,_ = eval_policy(policy, args.env,args.seed+np.random.randint(0,5),eval_episodes=2,logger=logger) 
                logger.store(TestMPCEvaluation=test_mpc_eval)

            evaluation_rewards, evaluation_episodes, evaluation_costs = 0, 0, 0

            logger.log_tabular('Timesteps', total_timesteps)
            
            if 'pen' in args.env: # Special environment specific evaluation for claw and pen environments
                if 'explore' in args.policy:
                    logger.log_tabular('TestSuccessRate', with_min_and_max=True)
                    logger.log_tabular('TestSuccessPercentage',with_min_and_max=True)
                logger.log_tabular('SuccessRate', with_min_and_max=True)
                logger.log_tabular('SuccessPercentage', with_min_and_max=True)
            elif 'pddm' in args.env:
                logger.log_tabular('Score', with_min_and_max=True)
            
            if noise_amount!=0 or 'ARC' in args.policy:
                logger.log_tabular('TestMPCEvaluation', with_min_and_max=True)    
            logger.log_tabular('MPCEvaluation', with_min_and_max=True)
            logger.log_tabular('MPCCostEvaluation', with_min_and_max=True)
            logger.log_tabular('ActorEvaluation', with_min_and_max=True)
            logger.log_tabular('DynamicsTrainLoss', average_only=True)
            logger.log_tabular('DynamicsValLoss', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="LOOP_SAC_ARC")
    parser.add_argument("--env", default="MBRLHalfCheetah-v0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e3, type=int)
    parser.add_argument("--eval_freq", default=2e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--dynamics_freq", default=250, type=int)
    parser.add_argument("--exp_name", default="dump")
    parser.add_argument('--config', '-c', type=str, default='configs/config.yml', help="specify the path to the configuation file of the models")
    args = parser.parse_args()
    run_loop(args)