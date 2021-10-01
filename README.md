# LOOP: Learning Off-Policy with Online Planning
Accepted in CoRL 2021. 



## Install
- PyTorch 1.5
- OpenAI Gym
- [MuJoCo](https://www.roboti.us/license.html)
- tqdm 
- [D4RL dataset](https://github.com/rail-berkeley/d4rl)


## File Structure
- LOOP (Core method)
- - Training code (Online RL): `train_loop_sac.py`
- - Training code (Offline RL): `train_loop_offline.py`
- - Training code (safe RL): `train_loop_safety.py`
- - Policies (online/offline/safety): `policies.py` 
- - ARC/H-step lookahead policy: `controllers/`
- Environments: `envs/`
- Configurations: `configs/`

## Instructions
- All the experiments are to be run under the root folder. 
- Config files  in `configs/` are used to specify hyperparameters for controllers and dynamics.
- Please keep all the other values in yml files consistent with hyperparamters given in paper to reproduce the results in our paper.



## Experiments

### Sec 6.1 LOOP for Online RL
```
python train_loop_sac.py --env=<env_name> --policy=LOOP_SAC_ARC --exp_name=<location_to_logs> 
```
Environments wrappers with their termination condition can be found under `envs/`

### Sec 6.2 LOOP for Offline RL

Download CRR trained models from [Link](https://drive.google.com/drive/folders/1JxCaHpCNrSAdgmla0RwuUfQltzvnBP8z?usp=sharing) into the root folder.


```
python train_loop_offline.py --env=<env_name> --policy=LOOP_OFFLINE_ARC --exp_name=<location_to_logs>  --offline_algo=CRR --prior_type=CRR
```

Currently supported for d4rl MuJoCo locomotions tasks only.

### Sec 6.3 LOOP for Safe RL

```
python train_loop_safety.py --env=<env_name> --policy=safeLOOP_ARC --exp_name=<location_to_logs> 
```
Safety environments can be found under `envs/safety_envs.py`



## References
Parts of the codes are used from the references mentioned below:


@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}

