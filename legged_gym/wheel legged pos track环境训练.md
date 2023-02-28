# wheel-legged stand still环境训练记录



## 位置跟踪

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
位置跟踪reward为exp奖励

![image-20230228225956145](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228225956145.png)

位置跟踪reward为平方惩罚

![image-20230228231313073](http://hongxiwong-pic.oss-cn-beijing.aliyuncs.com/img/image-20230228231313073.png)
