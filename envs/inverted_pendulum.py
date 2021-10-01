import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch
class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        self._max_episode_steps=1000

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent



    def termination_function(self,obs, act, next_obs):
        if torch.is_tensor(next_obs):
            notdone = torch.isfinite(next_obs).all(dim=-1).float() * (torch.abs(next_obs[:,1]) <= .2)
        else:
            notdone = np.isfinite(next_obs).all(axis=-1) \
                    * (np.abs(next_obs[:,1]) <= .2)
            
        done = 1-notdone

        done = done[:,None]

        return done
