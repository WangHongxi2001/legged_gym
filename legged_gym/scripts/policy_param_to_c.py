from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import struct


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def export2c(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 36

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    # load policy
    policy = ppo_runner.get_actor_critic(device=env.device)
    weight = []
    bias = []
    print(policy.actor[0].weight)
    for i in range(len(policy.actor)):
        if i % 2 == 1:
            continue
        weight += (
            policy.actor[i]
            .weight.transpose(0, 1)
            .reshape(
                -1,
            )
            .to("cpu")
            .tolist()
        )
        #
        bias += policy.actor[i].bias.to("cpu").tolist()

    file = open(r"param.c", "w", encoding="UTF-8")
    file.write('#include "policy_param.h"\n\n')

    file.write(
        "uint32_t hidden_layers_size[%s] = {"
        % str(len(train_cfg.policy.actor_hidden_dims))
    )
    for index, value in enumerate(train_cfg.policy.actor_hidden_dims):
        if index == len(train_cfg.policy.actor_hidden_dims) - 1:
            file.write("%s};\n\n" % str(value))
        else:
            file.write("%s," % str(value))

    file.write("uint32_t weight_array[%s] = {" % str(len(weight)))
    for index, value in enumerate(weight):
        if index == len(weight) - 1:
            file.write("%s};\n\n" % float_to_hex(value))
        else:
            if (index + 1) % 5 == 0 and index != 0:
                file.write("%s,\n" % float_to_hex(value))
            else:
                file.write("%s," % float_to_hex(value))

    file.write("uint32_t bias_array[%s] = {" % str(len(bias)))
    for index, value in enumerate(bias):
        if index == len(bias) - 1:
            file.write("%s};" % float_to_hex(value))
        else:
            if (index + 1) % 5 == 0 and index != 0:
                file.write("%s,\n" % float_to_hex(value))
            else:
                file.write("%s," % float_to_hex(value))
    file.write("extern uint32_t bias_array[%s];" % str(len(bias)))
    file.write("extern uint32_t weight_array[%s];" % str(len(weight)))
    file.close()


if __name__ == "__main__":
    args = get_args()
    export2c(args)
