# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import matplotlib.pyplot as plt

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    return ppo_runner.alg.storage

def plot(storage):
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    plt.plot(storage.rewards_buf)
    #plt.xlabel("iterations times")
    plt.title("mean rewards")
    plt.grid(True)

    plt.subplot(2,3,2)
    plt.plot(storage.episode_length_buf)
    #plt.xlabel("iterations times")
    plt.title("mean episodes length")
    plt.grid(True)

    plt.subplot(2,3,3)
    plt.plot(storage.mean_kl_buf)
    #plt.xlabel("iterations times")
    plt.title("mean kl")
    plt.grid(True)

    plt.subplot(2,3,4)
    plt.plot(storage.mean_value_loss_buf)
    #plt.xlabel("iterations times")
    plt.title("mean value loss")
    plt.grid(True)

    plt.subplot(2,3,5)
    plt.plot(storage.mean_surrogate_loss_buf)
    #plt.xlabel("iterations times")
    plt.title("mean surrogat loss")
    plt.grid(True)

    plt.subplot(2,3,6)
    plt.plot(storage.mean_entropy_buf)
    #plt.xlabel("iterations times")
    plt.title("mean entropy")
    plt.grid(True)
    
    # plt.show()

    plt.figure(figsize=(9,6))
    plt.plot(storage.rew_lin_vel_tracking_buf)
    plt.xlabel("steps")
    plt.title("mean rew_lin_vel_tracking")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    args = get_args()
    storage = train(args)
    plot(storage)
