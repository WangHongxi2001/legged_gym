import numpy as np
class Attitude:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.quat = np.zeros((num_envs,4), dtype=float)
        self.yaw = np.zeros(num_envs, dtype=float)
        self.pitch = np.zeros(num_envs, dtype=float)
        self.roll = np.zeros(num_envs, dtype=float)
        self.acc = np.zeros((num_envs,3), dtype=float)
        self.gyro = np.zeros((num_envs,3), dtype=float)