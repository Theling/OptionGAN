import numpy as np
import tensorflow as tf
import gym

from pgbox.utils import *
# from pgbox.trpo.model import *
import argparse
from pgbox.trpo.rollouts import *
import json
from pgbox.parallel_algo_utils import *
from pgbox.policies.gaussian_mlp_policy import *
from irlbox.utils import load_expert_rollouts
from irlbox.irlgan.trainer import ParallelTrainer
from irlbox.discriminators.mlp_discriminator import MLPDiscriminator
from pgbox.valuefunctions.nn_vf import *

from pgbox.sampling_utils import apply_transformers

import joblib


parser = argparse.ArgumentParser(description='TRPO.')
# these parameters should stay the same
parser.add_argument("--task", type=str, default='Hopper-v2')
parser.add_argument("--expert_rollouts_path", type=str, default='./data/Hopper-v2_data_10_rollouts.pkl')
parser.add_argument("--num_expert_rollouts", type=int, default=10)
parser.add_argument("--timesteps_per_batch", type=int, default=25000)
parser.add_argument("--n_iters", type=int, default=500)
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--max_kl", type=float, default=.01)
parser.add_argument("--cg_damping", type=float, default=1e-1)
parser.add_argument("--num_threads", type=int, default=5)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--lam", type=float, default=.97)
parser.add_argument("--use_reward_filter", action="store_true", help="Turn of default of original TRPO code.")
parser.add_argument("--use_obs_filter", action="store_true", help="Turn of default of original TRPO code.")
parser.add_argument("--discriminator_lr", default=1e-3, type=float)
parser.add_argument("--log_dir", default="./logs/")
parser.add_argument("--discriminator_size", nargs="+", default=(128,128), type=int)
parser.add_argument("--policy_size", nargs="+", default=(128,128), type=int)
parser.add_argument("--concat_prev_timestep", action="store_true")
parser.add_argument("--use_ppo", action="store_true")
parser.add_argument("--d_l2_penalty_weight", default=0.0, type=float)

args = parser.parse_args()

# logger.add_text_output(args.log_dir + "debug.log")
logger.add_tabular_output(args.log_dir + "progress.csv")

learner_env = gym.make(args.task)

expert_rollouts = load_expert_rollouts(args.expert_rollouts_path, max_traj_len = -1, num_expert_rollouts = args.num_expert_rollouts)

args.transformers = None


print("Using discriminator of size")
print(args.discriminator_size)

policy = GaussianMLPPolicy(learner_env, hidden_sizes=args.policy_size, activation=tf.nn.tanh)
baseline = MLPValueFunction(learner_env, hidden_sizes=args.policy_size, activation=tf.nn.tanh)

print(args.use_ppo)

trpo = ParallelTRPO(learner_env, policy, args, vf=baseline)
discriminator = MLPDiscriminator(learner_env.observation_space.shape[0], hidden_sizes=args.discriminator_size, activation=tf.nn.tanh, learning_rate=args.discriminator_lr, l2_penalty_weight=args.d_l2_penalty_weight)

trainer = ParallelTrainer(trpo, expert_rollouts, discriminator, policy, args)

iterations = 0
while iterations <= 115:
    iterations = trainer.step()
    joblib.dump(trpo.policy_weights, os.path.join('./logs/', f'param_{iterations}.pkl'))
trainer.end()
