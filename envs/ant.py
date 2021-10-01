from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

class AntTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self._max_episode_steps=1000

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        cost = (abs(ob[0])-0.2)*(abs(ob[0])>=0.2)
        return ob, reward, done, dict(
            cost = cost,
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def termination_function(self,obs, act, next_obs):

        x = next_obs[:, 1]
        if torch.is_tensor(next_obs):
            not_done = torch.isfinite(next_obs).all(dim=-1).float()\
                         * (x >= .2).float() \
                        * (x<=1.0).float()
        else:
            not_done = 	np.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)

        done = 1-not_done
        done = done[:,None]
        return done

    def get_cost_trajectory(self, trajectory):
        """
        Reward function definition.
        """
        # print(observation.shape)
        # cost = np.any((abs(observation[:,0])>0.5))*5
        thres = 0.2
        if trajectory.ndim>2:
            if torch.is_tensor(trajectory):
                cost = (abs(trajectory[:,:,0])-thres)* (abs(trajectory[:,:,0])>=thres)
                # cost = (abs(trajectory[:,:,0])-thres)**2 *(abs(trajectory[:,:,0])>=thres)
                traj_cost = torch.sum(cost,dim=0)
                return traj_cost

            cost = (abs(trajectory[:,:,0])-thres)* (abs(trajectory[:,:,0])>=thres)
            traj_cost = np.sum(cost,axis=0)
            return traj_cost

        else:
            if torch.is_tensor(trajectory):
                traj_cost =  torch.sum((abs(trajectory[:,0])-thres)*(abs(trajectory[:,0])>=thres))
            else:
                traj_cost = np.sum((abs(trajectory[:,0])-thres)*(abs(trajectory[:,0])>=thres))               

        return traj_cost

# class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         mujoco_env.MujocoEnv.__init__(self, '%s/assets/hopper.xml' % dir_path, 4)
#         self.prev_qpos = None
 
#         # mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
#         utils.EzPickle.__init__(self)
#         self._max_episode_steps=1000

#     def step(self, a):
#         self.prev_qpos = self.sim.data.qpos[0]
#         posbefore = self.sim.data.qpos[0]
#         self.do_simulation(a, self.frame_skip)
#         posafter, height, ang = self.sim.data.qpos[0:3]

        
#         alive_bonus = 1.0
#         reward = (posafter - posbefore) / self.dt
#         reward += alive_bonus
#         reward -=3* (height-1.3)**2 
#         reward -= 0.1 * np.square(a).sum()
#         s = self.state_vector()
#         # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
#         #             (height > .7) and (abs(ang) < .2))
#         done = False
#         ob = self._get_obs()
#         info = {'redundant_reward':-3* (height-1.3)**2 }
#         return ob, reward, done, info

#     def _get_obs(self):
#         return np.concatenate([
#             (self.sim.data.qpos.flat[:1]-self.prev_qpos)/self.dt,
#             self.sim.data.qpos.flat[1:],
#             np.clip(self.sim.data.qvel.flat, -10, 10)
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
#         self.set_state(qpos, qvel)
#         self.prev_qpos = np.copy(self.sim.data.qpos[0])
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 2
#         self.viewer.cam.distance = self.model.stat.extent * 0.75
#         self.viewer.cam.lookat[2] = 1.15
#         self.viewer.cam.elevation = -20


# class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         self.prev_qpos = None
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#         mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
#         utils.EzPickle.__init__(self)
#         self._max_episode_steps=1000

#     def step(self, action):
#         self.prev_qpos = np.copy(self.sim.data.qpos.flat)
#         self.do_simulation(action, self.frame_skip)
#         ob = self.get_obs()

#         reward_ctrl = -0.1 * np.square(action).sum()
#         reward_run = ob[0] - 0.0 * np.square(ob[2])
#         reward = reward_run + reward_ctrl

#         done = False
#         return ob, reward, done, {}

#     def get_obs(self):
#         return np.concatenate([
#             (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
#             self.sim.data.qpos.flat[1:],
#             self.sim.data.qvel.flat,
#         ])

#     def reset_model(self):
#         qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
#         qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
#         self.set_state(qpos, qvel)
#         self.prev_qpos = np.copy(self.sim.data.qpos.flat)
#         return self.get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.25
#         self.viewer.cam.elevation = -55