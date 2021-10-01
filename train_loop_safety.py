from envs.safety_envs import SafetyGymEnv
import numpy as np
import torch
import gym
import argparse
import os
from policies import get_policy
import time
import sac
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

def eval_policy(policy, env_name, seed, eval_episodes=5):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    avg_cost = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        i=0
        while not done:
            i+=1
            action,_ = policy.get_action(np.array(state))
            next_state, reward, done, info = eval_env.step(action)
            state = next_state
            avg_reward += reward
            if 'cost' in info:
                avg_cost += info['cost']

    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_cost




DEFAULT_ENV_CONFIG_POINT = dict(
    action_repeat=5,
    max_episode_length=1000,
    use_dist_reward=True,
    stack_obs=False,
)
DEFAULT_ENV_CONFIG_CAR = dict(
    action_repeat=5,
    max_episode_length=1000,
    use_dist_reward=True,
    stack_obs=False,
)

rc_envs = ['Straight-v0','Circle-v0','Drift-v0']

def run_loop(args):
    config = load_config(args.config)
    logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(
        args.policy, args.env, args.seed))
    print("---------------------------------------")


    if 'PointGoal1' in args.env:
        env_config = DEFAULT_ENV_CONFIG_POINT
        env = SafetyGymEnv(robot='Point', task="goal", level=1, seed=args.seed, config=env_config)
        env_fn = lambda: SafetyGymEnv(robot='Point', task="goal", level=1, seed=args.seed, config=env_config)
    elif 'CarGoal1' in args.env:
        env_config = DEFAULT_ENV_CONFIG_CAR
        env = SafetyGymEnv(robot='Car', task="goal", level=1, seed=args.seed, config=env_config)
        env_fn = lambda: SafetyGymEnv(robot='Car', task="goal", level=1, seed=args.seed, config=env_config)
    else:
        env = gym.make(args.env)
        env_fn = lambda:gym.make(args.env)


    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("State dim: {}, Action dim: {}".format(state_dim,action_dim))


    # Create replay buffer
    replay_buffer = sac.ReplayBuffer(state_dim, action_dim,int(1e6))

    # Choose a controller
    policy, sac_policy, dynamics, lookahead_policies = get_policy(args,  env, replay_buffer, config, policy_name=args.policy, env_fn=env_fn)
    
    # Noise to be added to controller while executing trajectory
    noise_amount = config['mpc_config']['exploration_noise']

    start_time = time.time()
    total_timesteps = 0
    episode_count =0 
    episode_timesteps = 0
    episode_reward, episode_cost = 0, 0
    evaluation_rewards, evaluation_costs = 0, 0
    evaluation_episodes = 0
    state, done = env.reset(), False

    for t in range(int(args.max_timesteps)):
        total_timesteps += 1
        episode_timesteps += 1
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.get_action(np.array(state),env=env)
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
            done_bool = float(
                done) if episode_timesteps < env._max_episode_steps else 0

            if episode_timesteps >= env._max_episode_steps:
                done=True
        else:
            done_bool = float(
                done) if episode_timesteps < 1000 else 0

            if episode_timesteps >= 1000:
                done=True


        replay_buffer.store(state, action, reward,next_state,  done_bool, cost=info["cost"])
        state = next_state


        if args.policy in lookahead_policies :
            if t >= args.start_timesteps and t%sac_policy.update_every==0:
                sac_policy.train()


        if (t+1) % args.dynamics_freq == 0:
            dynamics_trainloss,dynamics_valloss = dynamics.train()
            logger.store(DynamicsTrainLoss = dynamics_trainloss, DynamicsValLoss = dynamics_valloss)


        if done:
            policy.reset()
            evaluation_costs += episode_cost
            evaluation_rewards += episode_reward
            logger.store(MPCEvaluation=evaluation_rewards, MPCCostEvaluation=evaluation_costs)
            episode_reward, episode_cost = 0, 0
            evaluation_rewards, evaluation_costs = 0,0
            evaluation_episodes += 1
            episode_count+=1
            state, done = env.reset(), False
            episode_timesteps = 0


        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if args.policy in lookahead_policies:
                if(config['sac_config']['evaluation_mode']=='actor') :
                    actor_rew = eval_policy_actor(sac_policy, args.env, args.seed+np.random.randint(0,5))
                    logger.store(ActorEvaluation=actor_rew)
                else:
                    logger.store(ActorEvaluation=0)

            evaluation_rewards, evaluation_episodes, evaluation_costs = 0, 0, 0
            logger.log_tabular('Timesteps', total_timesteps)
            logger.log_tabular('MPCEvaluation', with_min_and_max=True)
            logger.log_tabular('MPCCostEvaluation', with_min_and_max=True)
            logger.log_tabular('ActorEvaluation', with_min_and_max=True)
            logger.log_tabular('DynamicsTrainLoss', average_only=True)
            logger.log_tabular('DynamicsValLoss', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="safeLOOP_ARC")
    parser.add_argument("--env", default="PointGoal-v1")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    parser.add_argument("--eval_freq", default=3e3, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--dynamics_freq", default=250, type=int)
    parser.add_argument("--exp_name", default="dump")
    parser.add_argument('--config', '-c', type=str, default='configs/safety_config_gym.yml', help="specify the path to the configuation file of the models")
    args = parser.parse_args()
    run_loop(args)