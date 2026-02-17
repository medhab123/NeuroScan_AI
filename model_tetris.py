# tetris main environment
from datetime import datetime # for logs

import gymnasium as gym
from ale_py import ALEInterface

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor # does not work with VecFrameStack if placed afterwards
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper 
from stable_baselines3.common.vec_env import VecFrameStack 
from stable_baselines3.common.vec_env import VecNormalize

from tensorboard_video_recorder import TensorboardVideoRecorder

# notes:
# command to start up tensorboard videos: tensorboard --bind_all --port 8889 --logdir tetris_logs
# took "tensorboard_video_recorder" from E2

def main():

    # environment setup ---------

    experiment_name = "ppo_cnn_tetris_vec_wrapper_atari_wrapper" # modify as needed
    experiment_logdir = f"tetris_logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # create tetris environment
    # env = gym.make("ALE/Tetris-v5", render_mode="rgb_array") 
    
    # vec env + AtariWrapper + VecFrameStack to make things easier for the agent
    env = make_vec_env("ALE/Tetris-v5", n_envs=8, seed=0, wrapper_class=AtariWrapper)
    env = VecFrameStack(env, n_stack=4)
    
    # monitor wrap
    # env = Monitor(env)

    # video for Tensorboard
    video_trigger = lambda step: step % 2e4 == 0
    env = TensorboardVideoRecorder(env=env,
                                   video_trigger=video_trigger,
                                   video_length=2000,
                                   fps=30,
                                   record_video_env_idx=0,
                                   tb_log_dir=experiment_logdir)

    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.) # normalization #EDIT: not needed

    # model setup --------------
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=experiment_logdir)
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_tetris_pixels")

if __name__ == '__main__':
    main()