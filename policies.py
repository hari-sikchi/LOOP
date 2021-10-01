# Policy definitions for Online, Offline and Safe RL

from controllers import arc, arc_offline, arc_safety
import numpy as np
import gym
from models.model_PETS import EnsembleDynamicsModel
from models.predict_env_pets import PredictEnv as PredictEnvPETS
import sac
import torch

# Default termination function that outputs done=False
def default_termination_function(state,action,next_state):
    if (torch.is_tensor(next_state)):
        done = torch.zeros((next_state.shape[0],1))
    else:
        done = np.zeros((next_state.shape[0],1))
    return done


def get_policy(all_args, env, replay_buffer, config, policy_name='LOOP_SAC',env_fn=None):
    policy,sac_policy = None, None
    dynamics_config = config['dynamics_config']
    mpc_config = config['mpc_config']
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim= dynamics_config['hidden_dim']

    if all_args.policy == 'LOOP_SAC_ARC':
        env_model = EnsembleDynamicsModel(7, 5, state_dim, action_dim, 1, hidden_dim,
                                          use_decay=True)
        dynamics = PredictEnvPETS(env_model,replay_buffer, all_args.env, 'pytorch')     
        sac_policy = sac.SAC(lambda:gym.make(all_args.env), dynamics, replay_buffer, env.termination_function)
        sac_policy.update_every=50
        sac_policy.update_after=1000
        policy = arc.ARC(
            env,
            dynamics,
            sac_policy,env.termination_function)


    elif all_args.policy == 'LOOP_OFFLINE_ARC':
        env_model = EnsembleDynamicsModel(7, 5, state_dim, action_dim, 1, hidden_dim,
                                          use_decay=True)
        dynamics = PredictEnvPETS(env_model,replay_buffer, all_args.env, 'pytorch')       

        sac_policy = sac.SAC(lambda:gym.make(all_args.env),replay_buffer)
        sac_policy.update_every=50
        sac_policy.update_after=1000
        if hasattr(env, 'termination_function:'):
            policy = arc_offline.ARC_offline(
                env,
                dynamics,
                sac_policy,env.termination_function)

        else:
            policy = arc_offline.ARC_offline(
                env,
                dynamics,
                sac_policy,default_termination_function)

    elif all_args.policy == 'safeLOOP_CEM':
        env_model = EnsembleDynamicsModel(7, 5, state_dim, action_dim, 1, hidden_dim,
                                          use_decay=True)
        dynamics = PredictEnvPETS(env_model,replay_buffer, all_args.env, 'pytorch')  
        sac_policy = sac.SAC(env_fn,dynamics,replay_buffer,default_termination_function)
        sac_policy.update_every=50
        sac_policy.update_after=1000
        policy = arc_safety.safeCEM(
            env,
            dynamics,
            sac_policy,default_termination_function)
    elif all_args.policy == 'safeLOOP_ARC':
        env_model = EnsembleDynamicsModel(7, 5, state_dim, action_dim, 1, hidden_dim,
                                          use_decay=True)
        dynamics = PredictEnvPETS(env_model,replay_buffer, all_args.env, 'pytorch')  
        sac_policy = sac.SAC(env_fn,dynamics,replay_buffer,default_termination_function)
        sac_policy.update_every=50
        sac_policy.update_after=1000
        policy = arc_safety.safeARC(
            env,
            dynamics,
            sac_policy,default_termination_function)

    if 'OFFLINE' not in all_args.policy and 'CEM' in all_args.policy:
        policy.horizon = mpc_config['horizon']
        policy.sol_dim = env.action_space.shape[0] * mpc_config['horizon']
        policy.ub = np.repeat(env.action_space.high,mpc_config['horizon'],axis=0)
        policy.lb = np.repeat(env.action_space.low,mpc_config['horizon'],axis=0)
        policy.mean = np.zeros((policy.sol_dim,))
        policy.N = mpc_config['CEM']['popsize']
        policy.mixture_coefficient = mpc_config['CEM']['mixture_coefficient']
        policy.particles = mpc_config['CEM']['particles']
        policy.max_iters = mpc_config['CEM']['max_iters']
        policy.num_elites = mpc_config['CEM']['num_elites']
        policy.alpha = mpc_config['CEM']['alpha']
        if 'reward_horizon' in mpc_config['CEM'].keys():
            policy.reward_horizon = mpc_config['CEM']['reward_horizon']
    elif 'OFFLINE' not in all_args.policy and 'ARC' in all_args.policy:
        policy.horizon = mpc_config['horizon']
        policy.sol_dim = env.action_space.shape[0] * mpc_config['horizon']
        policy.ub = np.repeat(env.action_space.high,mpc_config['horizon'],axis=0)
        policy.lb = np.repeat(env.action_space.low,mpc_config['horizon'],axis=0)
        policy.mean = np.zeros((policy.sol_dim,))
        policy.N = mpc_config['ARC']['popsize']
        policy.mixture_coefficient = mpc_config['ARC']['mixture_coefficient']
        policy.particles = mpc_config['ARC']['particles']
        policy.max_iters = mpc_config['ARC']['max_iters']
        policy.alpha = mpc_config['ARC']['alpha'] 
        policy.kappa = mpc_config['ARC']['kappa'] 

    lookahead_policies=['LOOP_SAC_ARC','LOOP_OFFLINE_ARC','safeLOOP_CEM','safeLOOP_ARC']

    return policy,sac_policy, dynamics, lookahead_policies
