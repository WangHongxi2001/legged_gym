import torch
class Attitude:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.gyro = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)