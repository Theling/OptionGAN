import numpy as np
import tensorflow as tf
import gym

from pgbox.utils import *
from pgbox.trpo.model import *
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
import tf_util


parser = argparse.ArgumentParser(description='TRPO.')
# these parameters should stay the same
parser.add_argument("--task", type=str, default='Hopper-v2')
parser.add_argument("--expert_rollouts_path", type=str, default='./data/Hopper-v2_data_10_rollouts.pkl')
parser.add_argument("--policy_size", nargs="+", default=(128,128), type=int)

args = parser.parse_args()

# logger.add_text_output(args.log_dir + "debug.log")
# logger.add_tabular_output(args.log_dir + "progress.csv")

num_rollouts = 10
env = gym.make(args.task)
def one_rollout(sess, env, file_):
    
    max_steps =  env.spec.timestep_limit
    print(max_steps)
    policy = GaussianMLPPolicy(env, hidden_sizes=args.policy_size, activation=tf.nn.tanh)
    tf_util.initialize()
    policy_params = joblib.load(file_)
    # print([x.name for x in policy.get_params()])
    policy.set_param_values(sess, policy_params)
    
    ret = []

    policy_fn = lambda x: policy.act(x, sess, eval = True)[0]

    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        returns = []
        rewards = []
        observations = []
        actions = []
        while not done:
            action= policy_fn(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            # print(done)
            rewards.append(r)
            totalr += r
            steps += 1
            if True:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        ret.append({'observations': np.array(observations),
                    'actions': np.array(actions),
                    'rewards': np.array(rewards),
                    'mean_return': np.mean(returns), 
                    'std_return': np.std(returns) })
    returns  = [ele['mean_return'] for ele in ret]
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return ret

results = []
with tf.Session() as sess:
    for i in range(1, 115, 2):
        results.append(one_rollout(sess, env, file_= f'./logs/exp2/param_{i}.pkl'))

joblib.dump(results, './logs/exp2/results.pkl')

