
# Implements ARC optimizers
import copy
import numpy as np
import scipy.stats as stats
import torch

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")


class ARC(object):
    def __init__(self, env, models, critic, termination_function):
        ###########
        # params
        ###########
        self.horizon = 3
        self.N = 100
        self.models = models
        self.env = copy.deepcopy(env)
        self.critic = critic
        self.mixture_coefficient = 0.05
        self.max_iters = 5
        self.actor_traj = int(self.mixture_coefficient*self.N)
        self.action_dim = self.env.action_space.shape[0]
        self.sol_dim = self.env.action_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.action_space.high,self.horizon,axis=0)
        self.lb = np.repeat(self.env.action_space.low,self.horizon,axis=0)
        self.alpha = 0.1
        self.mean = np.zeros((self.sol_dim,))
        self.termination_function = termination_function
        self.particles = 4
        self.kappa = 1   # Hyperparam search [1,5, 10]

    def reset(self):
        self.mean = np.zeros((self.sol_dim,))

    def get_action(self, curr_state,deterministic=False): 
        actor_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * (self.actor_traj)) # Size [actor_traj,state_dim]
        curr_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * ((self.N+self.actor_traj)*self.particles))
        curr_state = np.expand_dims(curr_state, axis=0)
        curr_state = np.repeat(
            curr_state,
            self.models.model.network_size,
            0)  # [numEnsemble, N+actor_traj,state_dim]

        # initial mean and var of the sampling normal dist
        self.mean[:-self.action_dim] = self.mean[self.action_dim:]
        self.mean[-self.action_dim:] = self.mean[-2*self.action_dim:-self.action_dim]
        mean = self.mean
        var = np.tile(np.square(self.env.action_space.high[0]-self.env.action_space.low[0]) / 16, [self.sol_dim])

        # Add trajectories using actions suggested by actors
        actor_trajectories = np.zeros((self.actor_traj,self.sol_dim))
        actor_state_m = torch.FloatTensor(actor_state).to(device)

        for h in range(self.horizon):
            actor_actions_m = self.critic.ac.act_batch(actor_state_m[:,1:],deterministic=True)
            actor_state_m = self.models.get_forward_prediction_random_ensemble_t(actor_state_m[:,1:],actor_actions_m)
            actor_trajectories[:,h*self.action_dim:(h+1)*self.action_dim]=actor_actions_m.detach().cpu().numpy()

        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

        t = 0
        while ((t < self.max_iters)):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            action_traj = (X.rvs(size=(self.N, self.sol_dim)) * np.sqrt(constrained_var) + mean).astype(np.float32)
            action_traj = np.concatenate((action_traj,actor_trajectories),axis=0)
            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj,self.particles,axis=0)

            # actions should be between -1 and 1
            action_traj = np.clip(action_traj, self.env.action_space.low[0], self.env.action_space.high[0])
            states = torch.from_numpy(np.expand_dims(curr_state.copy(),axis=0)).float().to(device)
            actions = np.repeat(np.expand_dims(action_traj,axis=0),self.models.model.network_size,axis=0)
            actions = torch.FloatTensor(actions).to(device)

            for h in range(self.horizon):
                states_h = states[h,:,:,1:]
                next_states = self.models.get_forward_prediction_t(states_h, actions[:,:, h * self.action_dim:(h + 1) * self.action_dim])
                states = torch.cat((states,next_states.unsqueeze(0)), axis=0)
            states = states.cpu().detach().numpy()

            done = np.zeros((states.shape[1],states.shape[2],1)) # Shape [Ensembles, (actor_traj+N)*particles,1]
            # Set the reward of terminated states to zero
            for h in range(1,self.horizon+1):
                for ens in range(states.shape[1]):
                    not_done = 1-done[ens,:,:]
                    states[h,ens,:,0]*= not_done.astype(np.float32).reshape(-1)
                    done[ens,:,:] = np.logical_or(done[ens,:,:], self.termination_function(None,None,states[h,ens,:,1:]))
            

            # Terminal value function 
            returns = np.zeros((self.N+self.actor_traj,))
            actions_H = torch.from_numpy(action_traj[:, (self.horizon - 1) * self.action_dim:(
                self.horizon) * self.action_dim].reshape((self.N+self.actor_traj)*self.particles, -1)).float().to(device)
            actions_H = actions_H.repeat(self.models.model.network_size, 1)  
            # actions_H = actions_H.repeat_interleave(repeats=states.shape[1],dim=0)
            states_H = torch.from_numpy(
                    states[self.horizon-1, :, :, 1:].reshape(self.models.model.network_size*(self.N+self.actor_traj)*self.particles, -1)).float().to(device)
            terminal_q_rewards = self.critic.ac.q1(
                    states_H, actions_H).cpu().detach().numpy() 
            terminal_q_rewards = terminal_q_rewards.reshape(self.models.model.network_size,-1)
            # Trajectory costing
            for ensemble in self.models.model.elite_model_idxes:
                done[ensemble,:,:] = np.logical_or(done[ensemble,:,:],self.termination_function(None,None,states[self.horizon-1,ensemble,:,1:]))
                not_done = 1-done[ensemble,:,:]
                q_rews = terminal_q_rewards[ensemble,:]*not_done.reshape(-1)
                n = np.arange(0,self.N+self.actor_traj,1).astype(int)
                for particle in range(self.particles):
                    returns[n]+= np.sum(states[:self.horizon, ensemble, n*self.particles+particle, 0],axis=0) + q_rews.reshape(-1)[n*self.particles+particle]

            returns /= (len(self.models.model.elite_model_idxes)*self.particles)
            max_reward = np.max(returns)
            score = np.exp(self.kappa*(returns-max_reward))            
            score/= np.sum(score)
            
            mean = np.sum(action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:]*score.reshape(-1,1),axis=0)/(np.sum(score)+1e-10)
            new_var = np.average((action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:]-mean)**2, weights=score.reshape(-1),axis=0)
            var =  (self.alpha) * var + (1 - self.alpha) * new_var            
            t += 1

        if deterministic:
            self.mean= mean
        else:
            self.mean = action_traj[np.random.choice(np.arange(score.shape[0]),p=score/np.sum(score))*self.particles,:]

        # changing to old mean
        # return mean[:self.action_dim]

        return self.mean[:self.action_dim]

