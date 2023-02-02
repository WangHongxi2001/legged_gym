import torch

from .wheel_legged_robot_config import WheelLeggedRobotCfg

class Leg:
    def __init__(self, cfg: WheelLeggedRobotCfg.Leg, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.left_index = 0
        self.right_index = 1
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        self.b = cfg.offset
        self.l1 = cfg.leg0_length
        self.l2 = cfg.leg1_length

        self.L0_kp = cfg.Ctrl.L0_kp
        self.L0_kd = cfg.Ctrl.L0_kd
        self.L0_ff = cfg.Ctrl.L0_ff
        self.alpha_kp = cfg.Ctrl.alpha_kp
        self.alpha_kd = cfg.Ctrl.alpha_kd

        self.end_x = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.end_y = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.end_x_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.end_y_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.L0 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.L0_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta0 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta0_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.alpha_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.theta1 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta2 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta1_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta2_dot = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

    def Solve(self):
        self.end_x,self.end_y,self.L0,self.theta0 = self.Forward_Kinematics(self.theta1,self.theta2)
        dt = 0.001
        temp_end_x,temp_end_y,temp_L0,temp_theta0 = self.Forward_Kinematics(self.theta1+self.theta1_dot*dt,self.theta2+self.theta2_dot*dt)
        self.end_x_dot = (temp_end_x - self.end_x) / dt
        self.end_y_dot = (temp_end_y - self.end_y) / dt
        self.L0_dot = (temp_L0 - self.L0) / dt
        self.theta0_dot = (temp_theta0 - self.theta0) / dt
        self.alpha = self.theta0 - self.pi / 2 * torch.ones_like(self.alpha)
        self.alpha_dot = self.theta0_dot


    def Forward_Kinematics(self, theta1, theta2):
        end_x = self.b * torch.ones_like(theta1) + self.l1 * torch.cos(theta1) + self.l2 * torch.cos(theta1+theta2)
        end_y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta1+theta2)
        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x)
        return end_x, end_y, L0, theta0

    def VMC(self, F, T):
        T1 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        T2 = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        t11 = self.l1*torch.sin(self.theta0-self.theta1)-self.l2*torch.sin(self.theta1+self.theta2-self.theta0)

        t12 = self.l1*torch.cos(self.theta0-self.theta1)-self.l2*torch.cos(self.theta1+self.theta2-self.theta0)
        t12 = t12/self.L0

        t21 = -self.l2*torch.sin(self.theta1+self.theta2-self.theta0)

        t22 = -self.l2*torch.cos(self.theta1+self.theta2-self.theta0)
        t22 = t22/self.L0
        
        T1 = t11*F + t12*T
        T2 = t21*F + t22*T

        return T1, T2

    def PD_Update(self, L0_reference, alpha_reference):
        F = self.L0_kp*(L0_reference-self.L0) + self.L0_kd*(0-self.L0_dot) + self.L0_ff
        T = self.alpha_kp*(alpha_reference-self.alpha) + self.alpha_kd*(0-self.alpha_dot)
        return self.VMC(F,-T)

