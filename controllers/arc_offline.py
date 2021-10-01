
import copy
import numpy as np
import scipy.stats as stats
import torch

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")



class ARC_offline(object):
    def __init__(self, env, models, critic, termination_function):

        ###########
        # params
        ###########
        self.horizon = 5
        self.models = models
        self.env = copy.deepcopy(env)
        self.critic = critic
        self.max_iters = 1
        self.actor_traj = 0
        self.action_dim = self.env.action_space.shape[0]
        self.sol_dim = self.env.action_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.action_space.high,self.horizon,axis=0)
        self.lb = np.repeat(self.env.action_space.low,self.horizon,axis=0)
        self.alpha = 0.1
        self.mean = np.zeros((self.sol_dim,))
        self.termination_function=termination_function
        self.particles = 4
        self.N = 100
        self.kappa = 0.3
        self.beta = 0.0
        self.sigma = 0.01
        self.beta_pessimism = 1.0
        self.pessimistic_reward = True
        self.bc_prior = None
        self.prev_traj = None
        self.prior_type = 'CRR'
        self.offline_policy_type = 'CRR' 

    def reset(self):
        self.mean = np.zeros((self.sol_dim,))

    def reinitialize(self,horizon,kappa,sigma,beta,beta_pessimism):
        ###########
        # params
        ###########
        self.horizon = horizon
        self.sol_dim = self.env.action_space.shape[0] * self.horizon
        self.ub = np.repeat(self.env.action_space.high,self.horizon,axis=0)
        self.lb = np.repeat(self.env.action_space.low,self.horizon,axis=0)
        self.mean = np.zeros((self.sol_dim,))
        self.kappa = kappa
        self.beta = beta
        self.sigma = sigma
        self.beta_pessimism = beta_pessimism
        self.prev_traj = None
   
    def prior_act_batch(self, actor_state):
        if self.prior_type=='random':
            actor_actions = self.bc_prior.act_batch(actor_state)
        else:
            actor_actions = self.critic.ac.act_batch(actor_state,deterministic=True)
        return actor_actions

    def get_offline_q(self,state,action):
        if(self.offline_policy_type=='PLAS'):
            q_values = self.critic.ac.critic.q1(state,action)
        else:
            q_values = self.critic.ac.q1(state, action)
        return q_values

    def get_action(self, curr_state):
        actor_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * (self.N))# Size [actor_traj,state_dim]
        curr_state = np.array([np.concatenate(([0],curr_state.copy()),axis=0)] * ((self.N+self.actor_traj)*self.particles))
        curr_state = np.expand_dims(curr_state, axis=0)
        curr_state = np.repeat(
            curr_state,
            self.models.model.network_size,
            0)  # [numEnsemble, N+actor_traj,state_dim]

        # initial mean and var of the sampling normal dist
        self.mean[:-self.action_dim] = self.mean[self.action_dim:]
        self.mean[-self.action_dim:] = np.zeros((self.action_dim,))
        mean=self.mean
        sigma_ = np.tile(self.sigma, [self.sol_dim])

        actor_trajectories = np.zeros((self.N,self.sol_dim))
        actor_state = torch.FloatTensor(actor_state).to(device)
        for h in range(self.horizon):
            actor_actions_m = self.prior_act_batch(actor_state[:,1:])
            # actor_actions_m = self.critic.ac.act_batch(actor_state[:,1:])
            actor_state = self.models.get_forward_prediction_random_ensemble_t(actor_state[:,1:],actor_actions_m)
            actor_trajectories[:,h*self.action_dim:(h+1)*self.action_dim]=actor_actions_m.detach().cpu().numpy()


        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale= np.ones_like(mean))

        t = 0
        while ((t < 1)):
            noise = np.random.normal(loc=0, scale=1.0, size=(self.N, self.sol_dim)) * sigma_
            action_traj = noise.copy() + actor_trajectories
            for i in range(self.horizon):
                # Optional action smoothing
                if self.prev_traj is not None and self.horizon!=1:
                    if(i<self.horizon-1):
                        action_traj[:, i*self.action_dim:(i+1)*self.action_dim] = (1-self.beta)*action_traj[:,i*self.action_dim:(i+1)*self.action_dim]+self.beta*self.prev_traj[(i)*self.action_dim:(i+1)*self.action_dim].reshape(1,-1) 
                    else:
                        action_traj[:, i*self.action_dim:(i+1)*self.action_dim] = (1-self.beta)*action_traj[:,i*self.action_dim:(i+1)*self.action_dim]+self.beta*self.prev_traj[(i-1)*self.action_dim:(i)*self.action_dim].reshape(1,-1) 
                
            # Multiple particles go through the same action sequence
            action_traj = np.repeat(action_traj,self.particles,axis=0)
            action_traj = np.clip(action_traj, self.env.action_space.low[0], self.env.action_space.high[0])
            states = torch.from_numpy(np.expand_dims(curr_state.copy(),axis=0)).float().to(device)
            actions = np.repeat(np.expand_dims(action_traj,axis=0),self.models.model.network_size,axis=0)
            actions = torch.FloatTensor(actions).to(device)
            
            
            for h in range(self.horizon):
                states = torch.cat((states,self.models.get_forward_prediction_t(states[h,:,:,1:], actions[:,:, h * self.action_dim:(h + 1) * self.action_dim]).unsqueeze(0)), axis=0)
            states = states.cpu().detach().numpy()


            done = np.zeros((states.shape[1],states.shape[2],1)) # Shape [Ensembles, (actor_traj+N)*particles,1]
            # Set the reward of terminated states to zero
            for h in range(1,self.horizon+1):
                for ens in range(states.shape[1]):
                    done[ens,:,:] = np.logical_or(done[ens,:,:],self.termination_function(None,None,states[h,ens,:,1:]))
                    not_done = 1-done[ens,:,:]
                    states[h,ens,:,0]*=not_done.astype(np.float32).reshape(-1)
            
            # Find average cost of each trajectory
            returns = np.zeros((self.N+self.actor_traj,))
            particle_costs = np.zeros((self.N+self.actor_traj,self.particles))

            actions_H = torch.from_numpy(action_traj[:, (self.horizon - 1) * self.action_dim:(
                self.horizon) * self.action_dim].reshape((self.N+self.actor_traj)*self.particles, -1)).float().to(device)
            actions_H = actions_H.repeat_interleave(repeats=states.shape[1],dim=0)
            states_H = torch.from_numpy(
                    states[self.horizon-1, :, :, 1:].reshape((self.N+self.actor_traj)*self.particles*states.shape[1], -1)).float().to(device)
            
            terminal_q_rewards = self.get_offline_q(states_H, actions_H).cpu().detach().numpy()
            terminal_q_rewards = terminal_q_rewards.reshape(states.shape[1],-1)
            for ensemble in self.models.model.elite_model_idxes:
                done[ensemble,:,:] = np.logical_or(done[ensemble,:,:],self.termination_function(None,None,states[self.horizon-1,ensemble,:,1:]))
                not_done = 1-done[ensemble,:,:]
                q_rews = terminal_q_rewards[ensemble,:]*not_done.reshape(-1)
                n = np.arange(0,self.N+self.actor_traj,1).astype(int)
                for particle in range(self.particles):
                    if self.pessimistic_reward:
                        particle_costs[:,particle] = np.sum(states[:self.horizon, ensemble, n*self.particles+particle, 0],axis=0) + q_rews.reshape(-1)[n*self.particles+particle]
                        if (particle==self.particles-1):
                            returns[n]= np.mean(particle_costs,axis=1)-self.beta_pessimism*np.std(particle_costs,axis=1)
                    else:
                        returns[n]+= np.sum(states[:self.horizon, ensemble, n*self.particles+particle, 0],axis=0) + q_rews.reshape(-1)[n*self.particles+particle]



            returns /= (states.shape[1]*self.particles)
            max_reward = np.max(returns)
            score = np.exp(self.kappa*(returns-max_reward))
            mean = np.sum(action_traj[np.arange(0,self.N+self.actor_traj,1).astype(int)*self.particles,:]*score.reshape(-1,1),axis=0)/(np.sum(score)+1e-10)
            t += 1
          
        self.mean = mean
        self.prev_traj = self.mean.copy()
        
        return mean[:self.action_dim]
