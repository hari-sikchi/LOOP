from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        self._max_episode_steps=1000

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5





    def termination_function(self,obs, act, next_obs):
        if torch.is_tensor(next_obs):
            done = torch.tensor([False]).repeat(next_obs.shape[0]).view(-1,1)
        else:
            done = np.array([False]).repeat(next_obs.shape[0])
            done = done[:,None]
        return done



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