# wheel-legged stand still环境训练记录

## 使用新debug图像

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12：

![image-20230226164209690](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230226164209690.png)

## 换用 tanh 与更小学习率

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-4
activation = 'tanh'

![image-20230226201418315](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230226201418315.png)

## 添加 override_com

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = True

![image-20230227200716574](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227200716574.png)

## 修改驱动轮碰撞模型为圆柱

以下为修改驱动轮碰撞模型为圆柱后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False

![image-20230227202723804](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227202723804.png)

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = True

![image-20230227205649314](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227205649314.png)

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False

![image-20230227210846277](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227210846277.png)

## 修改 optim.Adam epsilon 参数为1e*5

以下为修改optim.Adam epsilon 参数为1e*5后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False

![image-20230227223102304](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227223102304.png)

## 修改 std 为 logstd

以下为修改 std 为 logstd后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False

![image-20230227232530862](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227232530862.png)

## 添加orthogonal_init

以下为添加orthogonal_init后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True

orthogonal_init会导致 entropy 无法保持下降趋势

![image-20230227233609006](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227233609006.png)

## 添加anneal_lr

以下为添加anneal_lr后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = False
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True
anneal_lr = True

应当配合learning_rate调整，搞不好有bug

![image-20230227235249163](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230227235249163.png)

![image-20230228000256657](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228000256657.png)

## 添加 early stop

以下为添加early stop后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = True
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True
anneal_lr = False

效果不咋好，搞不好有bug

![image-20230228001047767](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228001047767.png)

## 添加 clip_observations

以下为添加clip_observations后：

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 100
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 12
learning_rate = 1.e-3
activation = 'tanh'
early_stop = True
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True
anneal_lr = False
clip_observations = 10

未进行 normalization 直接 clip 到 10 可能不太合适，影响最终学习效果

![image-20230228205535150](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228205535150.png)

## 缩小num_steps_per_env

dt = 0.005, decimation = 2
num_envs = 4096
num_steps_per_env = 50
actor_hidden_dims = [8, 4, 2]
num_mini_batches = 6
learning_rate = 1.e-3
activation = 'tanh'
early_stop = True
override_com = False
replace_cylinder_with_capsule = False
orthogonal_init = True
anneal_lr = False
clip_observations = 100

![image-20230228221911378](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228221911378.png)

![image-20230228223857538](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228223857538.png)

修改为

num_envs = 4096 * 2
num_mini_batches = 6 * 2

效果反而更差

![image-20230228222444319](https://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228222444319.png)

修改为

num_envs = 4096
num_mini_batches = 6 * 2

也不咋好

![image-20230228223306833](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228223306833.png)

## 调整位置跟踪reward为exp奖励

torch.exp(-lin_pos_error/self.cfg.rewards.tracking_sigma)

![image-20230228230934669](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228230934669.png)

