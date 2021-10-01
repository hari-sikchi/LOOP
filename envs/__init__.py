from gym.envs.registration import register



register(
    id='MBRLHalfCheetah-v0',
    entry_point='envs.half_cheetah:HalfCheetahEnv'
)

register(
    id='MBRLInvertedPendulum-v0',
    entry_point='envs.inverted_pendulum:InvertedPendulumEnv'
)

register(
    id='MBRLWalker-v0',
    entry_point='envs.walker2d:Walker2dEnv'
)
register(
    id='MBRLAnt-v0',
    entry_point='envs.ant:AntTruncatedObsEnv'
)

register(
    id='MBRLSwimmer-v0',
    entry_point='envs.swimmer:SwimmerEnv'
)

register(
    id='MBRLHopper-v0',
    entry_point='envs.hopper:HopperEnv'
)

register(
    id='MBRLHumanoid-v0',
    entry_point='envs.humanoid:HumanoidTruncatedObsEnv'
)
