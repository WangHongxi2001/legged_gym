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

from legged_gym.envs.base.base_config import BaseConfig

class WheelLeggedRobotCfg(BaseConfig):
    class Leg:
        offset = 0.054
        leg0_length = 0.15
        leg1_length = 0.25

        class Ctrl:
            L0_kp = 1000
            L0_kd = 100
            L0_ff = 43

            alpha_kp = 250
            alpha_kd = 10

    class Wheel:
        radius = 0.0675
    class env:
        num_envs = 4096#*0+64
        num_observations = 34#-9
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 6
        env_spacing = 1.5  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh wheel_legged_tarrain
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 1 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 20.
        terrain_width = 20.
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # num_rows= 10 # number of terrain rows (levels)
        # num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, discrete]
        terrain_proportions = [0.4, 0.4, 0.2]
        num_rows= 3 # number of terrain rows (levels)
        num_cols = 5 # number of terrain cols (types)
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 2.0
        max_centripetal_accel = 5.0
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5 # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            wheel_vel = [-2.0, 2.0]
            wheel_vel_curriculum = 0
            ang_vel_z = [-3.0, 3.0]
            base_height = [0.15, 0.25]

    class init_state:
        pos = [0.0, 0.0, 0.1] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "lf0_Joint": 0., 
            "lf1_Joint": 0.95, 
            "l_Wheel_Joint":0.,
            "rf0_Joint": 0., 
            "rf1_Joint": -0.95,
            "r_Wheel_Joint":0.}

    class control:
        wheel_control_mode = 'Torque' # Torque, Velocity
        wheel_Velocity_Kp = 0.2
        action_scale_wheel_T = 1
        action_scale_wheel_Vel = 10

        leg_alpha_control_mode = 'Torque' # Torque, Position
        leg_alpha_Kp = 100
        leg_alpha_Kd = 4
        action_scale_leg_alpha_T = 1
        action_scale_leg_alpha_Pos = 0.1

        leg_L0_control_mode = 'Force' # Force, Position
        leg_L0_Kp = 1500
        leg_L0_Kd = 30
        action_scale_leg_L0_F = 5
        action_scale_leg_L0_Pos = 0.1
        
        action_offset_leg_L0_F = 42
        action_offset_leg_L0_Pos = 0.22

        stiffness = {
            'lf0_Joint': 0.0, 
            'lf1_Joint': 0.0, 
            'l_Wheel_Joint': 0.0, 
            'rf0_Joint': 0.0, 
            'rf1_Joint': 0.0, 
            'r_Wheel_Joint': 0.0}  # [N*m/rad]
        damping = {
            'lf0_Joint': 0.02, 
            'lf1_Joint': 0.02, 
            'l_Wheel_Joint': 0.0, 
            'rf0_Joint': 0.02, 
            'rf1_Joint': 0.02, 
            'r_Wheel_Joint': 0.0}     # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/wl/urdf/wl.urdf'
        name = "wheel_legged_robot"  # actor name
        foot_name = "Wheel" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = ['base_link', 'f0_Link', 'f1_Link']
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        override_inertia = False # 会引起 ValueError
        override_com = False
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        randomize_inertia = True
        randomize_inertia_scale = 0.1
        push_robots = True
        push_interval_s = 6
        max_push_vel_xy = 1.5
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.
        rand_force_curriculum_level = 0

    class rewards:
        class scales:
            # 50 1 10 15 400 3
            # 50 1 50 15 400 3 
            # 1 0.25
            termination = -0.0
            keep_balance = 5.0
            lin_vel_tracking = 1
            lin_vel_error_int_penalty = -0.1
            lin_vel_diff_penalty = -0.1*0
            ang_vel_z_tracking = 1.0
            ang_vel_x_penalty = -0.5
            ang_vel_y_penalty = -1.0
            pitch = -50.0#*0+1
            roll = -20.0#*0+1
            base_height_tracking = 2.0
            base_height_tracking_pb = 1.0*0
            base_height_dot_penalty = -0.1
            leg_theta = -0.5
            leg_theta_dot_penalty = -0.01
            leg_ang_diff = -2.0
            leg_ang_diff_dot_penalty = -0.05
            energy_penalty_T = -0.05
            energy_penalty_T_Leg = -0.001
            energy_penalty_F_Leg = -0.00005
            action_rate_wheel_T = -0.01
            action_rate_leg_alpha_T = -0.005
            action_rate_leg_alpha_F = -0.0005
            collision = -1.
            contacts_terminate_penalty = -10

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 10.
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        # obs_norm_std = [0.9700, 0.8448, 0.1063, 0.9589, 0.1073, 0.6086, 1.1977, 0.9319, 6.3768, 1.2439, 0.8284, 0.8145, 0.4152, 0.3502]
        obs_norm_std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        class obs_scales:
            wheel_motion = 1.0
            position = 1.0
            ang_vel = 1.0
            leg_theta = 1.0
            leg_theta_dot = 1.0
            leg_alpha = 1.0
            leg_alpha_dot = 1.0
            leg_L0 = 1.0
            leg_L0_dot = 1.0
            base_eular_angle = 1.0
            base_phi = 1.0
            base_phi_dot = 1.0
            base_pitch = 1.0
            base_roll = 1.0
            base_height = 1.0
            base_height_dot = 1.0
            gravity = 1.0
            lin_acc = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            reserve = 0.0
            wheel_motion = 0.1
            position = 0.01
            gravity = 0.05
            ang_vel = 0.3
            leg_alpha = 0.005
            leg_alpha_dot = 0.3
            leg_L0 = 0.005
            leg_L0_dot = 0.1
            
            # wheel_motion = 0.02
            # position = 0.01
            # gravity = 0.004
            # ang_vel = 0.03
            # leg_alpha = 0.005
            # leg_alpha_dot = 0.03
            # leg_L0 = 0.005
            # leg_L0_dot = 0.02

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class WheelLeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [48, 32, 24]
        critic_hidden_dims = [48, 32, 24]
        activation = 'tanh' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = True
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        early_stop = False
        anneal_lr = False
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1000 # number of policy updates
        observation_normalizing = False

        # logging
        save_interval = 25 # check for potential saves every this many iterations
        experiment_name = 'wheel_legged'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt