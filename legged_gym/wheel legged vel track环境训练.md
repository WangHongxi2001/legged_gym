# wheel-legged vel tracking环境训练记录



## wheel_control_mode = 'Torque'

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
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

![image-20230304202159673](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304202159673.png)

![image-20230304213627449](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304213627449.png)

num_steps_per_env = 50
num_mini_batches = 6

![image-20230304203333415](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304203333415.png)

num_steps_per_env = 75
num_mini_batches = 9

![image-20230304204304580](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304204304580.png)

![image-20230304211642373](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304211642373.png)

### replace_cylinder_with_capsule

replace_cylinder_with_capsule = True

num_steps_per_env = 75
num_mini_batches = 9

![image-20230304221509791](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304221509791.png)

### 将sim_tensor_process调整到step后

replace_cylinder_with_capsule = False
num_steps_per_env = 75
num_mini_batches = 9

![image-20230304233330583](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230304233330583.png)

replace_cylinder_with_capsule = True

![image-20230305140345966](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230305140345966.png)

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
