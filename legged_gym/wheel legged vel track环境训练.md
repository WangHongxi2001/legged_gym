# wheel-legged vel tracking环境训练记录

## wheel_control_mode = 'Torque'

### potential base

baseline：

termination = -0.0
keep_balance = 1.0
lin_vel_tracking = 2
lin_vel_error_int_penalty = -0.1
lin_vel_diff_penalty = -1.0
ang_vel_z_tracking = 1.0
ang_vel_x_penalty = -0.5
ang_vel_y_penalty = -1.0
pitch_penalty = -50.0
roll_penalty = -20.0
base_height_tracking = 3.0
base_height_dot_penalty = -0.1
leg_theta_penalty = -0.5
leg_theta_dot_penalty = -0.01
leg_ang_diff_penalty = -5.0
leg_ang_diff_dot_penalty = -0.1
energy_penalty_T = -0
energy_penalty_T_Leg = -0
energy_penalty_F_Leg = -0.0001
action_rate_wheel_T = -0
action_rate_leg_alpha_T = -0
action_rate_leg_alpha_F = -0
collision = -1.
contacts_terminate_penalty = -10

rew_ang_vel_z_tracking: 0.2664
rew_base_height_tracking: 0.6629
rew_lin_vel_tracking: 0.4350

![image-20230502173303456](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230502173303456.png)

termination = -0.0
keep_balance = 1.0
lin_vel_tracking = 2
lin_vel_error_int_penalty = -0.1
lin_vel_diff_penalty = -1.0
ang_vel_z_tracking = 1.0
pitch = -50.0*0+1
roll = -20.0*0+1
base_height_tracking = 3.0
leg_theta_penalty = -0.5
leg_theta_dot_penalty = -0.01
leg_ang_diff_penalty = -5.0
energy_penalty_F_Leg = -0.0001
collision = -1.
contacts_terminate_penalty = -10

## wheel_control_mode = 'Velocity'

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 75
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 9
learning_rate = 1.e-3
activation = 'tanh'
orthogonal_init = True
clip_observations = 100

termination = -0.0
lin_vel_tracking = 1.0
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
energy_penalty_T = -0.001*0
collision = -1.

![image-20230304214648935](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304214648935.png)



## curriculum

![image-20230804150443402](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230804150443402.png)
