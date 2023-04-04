# wheel-legged vel tracking环境训练记录



## wheel_control_mode = 'Torque'

### 加入critic_layers正交初始化

curriculum = True
observation_normalizing = False

wheel_vel_delta达到2.0

std: tensor([0.9569, 6.0264, 0.1030, 0.9743, 0.1058, 0.6129, 1.2044, 0.9214, 6.4278, 1.2449, 0.9841, 0.8261, 0.4296, 0.3502], device='cuda:0')

![image-20230307212728793](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230307212728793.png)

![image-20230307212737894](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230307212737894.png)

### 网络输入position改为vel_error_int

curriculum = True
observation_normalizing = False

lin_vel_tracking = 1.0
lin_pos_tracking = 1.0
ang_vel_z_tracking = 1.0
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
base_phi_penalty = -100.0
leg_ang_diff_penalty = -0.5
leg_ang_diff_dot_penalty = -0.1
collision = -1.

lin_pos_error = torch.square(self.commands[:,1] - self.Velocity.position[:])

wheel_vel_delta达到2.0

std: tensor([0.9700, 0.8448, 0.1063, 0.9589, 0.1073, 0.6086, 1.1977, 0.9319, 6.3768, 1.2439, 0.8284, 0.8145, 0.4152, 0.3502], device='cuda:0')

![image-20230308120114553](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230308120114553.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230308120127709.png" alt="image-20230308120127709" style="zoom:50%;" />

修改网络结构

lin_vel_tracking = 1.0
lin_pos_tracking = 1.0
ang_vel_z_tracking = 0.5
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
base_phi_penalty = -10.0
base_phi_dot_penalty = -0.1
leg_ang_diff_penalty = -0.5
leg_ang_diff_dot_penalty = -0.1
collision = -1.

self.Velocity.forward_error_int.view(self.num_envs,1) * self.obs_scales.position,
lin_pos_error = torch.square(self.commands[:,1] - self.Velocity.position[:])

wheel_vel_delta达到1.5

![image-20230310002753354](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230310002753354.png)

<img src="https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230310002802947.png" alt="image-20230310002802947" style="zoom:50%;" />

base_phi_penalty = -25.0

wheel_vel_delta达到1.5

![image-20230310003900600](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230310003900600.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230310003919298.png" alt="image-20230310003919298" style="zoom:50%;" />



### lr schedule 值得分析

lin_vel_tracking = 1.0
ang_vel_z_tracking = 0.5
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
base_phi_penalty = -25.0
base_phi_dot_penalty = -0.1
leg_ang_diff_penalty = -0.5
leg_ang_diff_dot_penalty = -0.1
collision = -1.

learning_rate = 1.e-3 #5.e-4
schedule = 'adaptive'

![image-20230313165908520](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230313165908520.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230313165919959.png" alt="image-20230313165919959" style="zoom: 33%;" />

learning_rate = 1.e-3 #5.e-4
schedule = 'fixed'

![image-20230313171215598](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230313171215598.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230313171224786.png" alt="image-20230313171224786" style="zoom:33%;" />

lin_vel_tracking = 1.0
ang_vel_z_tracking = 0.5
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
base_phi_penalty = -25.0
base_phi_dot_penalty = -2.5
leg_ang_diff_penalty = -0.5
leg_ang_diff_dot_penalty = -0.1
collision = -1.

learning_rate = 1.e-3 #5.e-4
schedule = 'adaptive'

actor_hidden_dims = [32, 16, 8]

![image-20230316124011840](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230316124011840.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230316124025659.png" alt="image-20230316124025659" style="zoom:50%;" />

### leg_alpha_control_mode = 'Mix'

leg_alpha_control_mode = 'Mix'

![image-20230316125131873](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230316125131873.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230316125144525.png" alt="image-20230316125144525" style="zoom:50%;" />

resampling_time = 10*0.8

![image-20230317173646223](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230317173646223.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230317173658655.png" alt="image-20230317173658655" style="zoom:50%;" />

### 删除cmd中位置

commit

<img src="https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230318185759345.png" alt="image-20230318185759345" style="zoom:50%;" />

<img src="https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230318185820816.png" alt="image-20230318185820816" style="zoom:50%;" />

### 完整

num_steps_per_env = 50
num_mini_batches = 5
desired_kl = 0.01

![image-20230403164124530](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403164124530.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403164131506.png" alt="image-20230403164131506" style="zoom:50%;" />

num_steps_per_env = 50+25
num_mini_batches = 5
desired_kl = 0.01

![image-20230403165630627](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403165630627.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403155957801.png" alt="image-20230403155957801" style="zoom:50%;" />

num_steps_per_env = 50+25
num_mini_batches = 9
desired_kl = 0.01

![image-20230403160910864](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403160910864.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403160917736.png" alt="image-20230403160917736" style="zoom:50%;" />

num_steps_per_env = 50+25
num_mini_batches = 5
desired_kl = 0.01*0.5

![image-20230403170644644](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403170644644.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403170656848.png" alt="image-20230403170656848" style="zoom:50%;" />

num_steps_per_env = 50+25
num_mini_batches = 9
desired_kl = 0.01
num_envs = 4096*2

![image-20230403171912891](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403171912891.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403171922242.png" alt="image-20230403171922242" style="zoom:50%;" />

orientation_y_penalty = -50.0

![image-20230403173032877](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403173032877.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403173022793.png" alt="image-20230403173022793" style="zoom:50%;" />

num_steps_per_env = 100
num_mini_batches = 10
desired_kl = 0.01
num_envs = 4096
orientation_y_penalty = -100

![image-20230403213227261](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403213227261.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403213235876.png" alt="image-20230403213235876" style="zoom:50%;" />

num_steps_per_env = 100
num_mini_batches = 5
desired_kl = 0.01*0.5
num_envs = 4096
orientation_y_penalty = -100

![image-20230403214524089](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403214524089.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403214514045.png" alt="image-20230403214514045" style="zoom:50%;" />

num_steps_per_env = 100
num_mini_batches = 5
desired_kl = 0.01
num_envs = 4096
orientation_y_penalty = -100

![image-20230403232323545](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403232323545.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403232330123.png" alt="image-20230403232330123" style="zoom:50%;" />

num_steps_per_env = 100
num_mini_batches = 5
desired_kl = 0.01
num_envs = 4096
orientation_y_penalty = -100
actor_hidden_dims = [64, 48, 32]

![image-20230403235257673](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403235257673.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230403235303361.png" alt="image-20230403235303361" style="zoom:50%;" />

在上一个参数基础上contacts_terminate_penalty = -10

![image-20230404000433477](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230404000433477.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230404000441027.png" alt="image-20230404000441027" style="zoom:50%;" />

replace_cylinder_with_capsule = False 并采用速度fifo
lin_vel_diff_penalty = -1.0

![image-20230404153355001](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230404153355001.png)

<img src="http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230404153406165.png" alt="image-20230404153406165" style="zoom:50%;" />

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
