# wheel-legged vel tracking环境训练记录



## 速度跟踪

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 50
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 6
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True
anneal_lr = False
clip_observations = 100

termination = -0.0
lin_vel_penalty = -0.1
lin_pos_tracking = 1.0
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
energy_penalty_T = -0.1
collision = -1.
stand_still = 1.

位置跟踪reward为exp奖励

实际跟踪效果较差

![image-20230301222803712](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230301222803712.png)

termination = -0.0
lin_vel_tracking = 1.0
lin_pos_tracking = 1.0
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
energy_penalty_T = -0.1
collision = -1.
stand_still = 1.

实际跟踪效果略好，但还不够好

![image-20230301223359135](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230301223359135.png)

![image-20230301224316383](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230301224316383.png)

修改 num_steps_per_env = 25
num_mini_batches = 3

![image-20230301225440064](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230301225440064.png)

考虑到100多次迭代后kl有一个巨大的peek，并且导致rewards显著下降，故使能early_stop：

看起来early stop也没啥调用，不如关了

![image-20230301225953884](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230301225953884.png)

lin_vel_tracking = 1.0
leg_theta_penalty = -1
leg_theta_dot_penalty = -0.1
collision = -1.
randomize_friction = True
friction_range = [0.5, 1.25]
randomize_base_mass = True
added_mass_range = [-1., 1.]
push_robots = True
push_interval_s = 7
max_push_vel_xy = 1.

![image-20230303154105877](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230303154105877.png)

randomize_friction = False
randomize_base_mass = False
push_robots = False

![image-20230303154804573](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230303154804573.png)

转为速度控制

![image-20230303155135157](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230303155135157.png)

randomize_friction = True
friction_range = [0.5, 1.25]
randomize_base_mass = True
added_mass_range = [-1., 1.]
push_robots = True

![image-20230303155750658](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230303155750658.png)
