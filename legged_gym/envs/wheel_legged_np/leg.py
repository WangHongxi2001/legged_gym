import numpy as np

from .wheel_legged_robot_config_np import WheelLeggedRobotCfg_np

class Leg:
    def __init__(self, cfg: WheelLeggedRobotCfg_np.Leg, num_envs):
        self.num_envs = num_envs

        self.left_index = 0
        self.right_index = 1

        self.b = cfg.offset
        self.l1 = cfg.leg0_length
        self.l2 = cfg.leg1_length

        self.L0_kp = cfg.Ctrl.L0_kp
        self.L0_kd = cfg.Ctrl.L0_kd
        self.L0_ff = cfg.Ctrl.L0_ff
        self.alpha_kp = cfg.Ctrl.alpha_kp
        self.alpha_kd = cfg.Ctrl.alpha_kd

        self.end_x = np.zeros((self.num_envs,2), dtype=float)
        self.end_y = np.zeros((self.num_envs,2), dtype=float)
        self.end_x_dot = np.zeros((self.num_envs,2), dtype=float)
        self.end_y_dot = np.zeros((self.num_envs,2), dtype=float)
        self.L0 = np.zeros((self.num_envs,2), dtype=float)
        self.L0_dot = np.zeros((self.num_envs,2), dtype=float)
        self.theta0 = np.zeros((self.num_envs,2), dtype=float)
        self.theta0_dot = np.zeros((self.num_envs,2), dtype=float)
        self.alpha = np.zeros((self.num_envs,2), dtype=float)
        self.alpha_dot = np.zeros((self.num_envs,2), dtype=float)

        self.theta1 = np.zeros((self.num_envs,2), dtype=float)
        self.theta2 = np.zeros((self.num_envs,2), dtype=float)
        self.theta1_dot = np.zeros((self.num_envs,2), dtype=float)
        self.theta2_dot = np.zeros((self.num_envs,2), dtype=float)

    def Solve(self):
        self.end_x,self.end_y,self.L0,self.theta0 = self.Forward_Kinematics(self.theta1,self.theta2)
        dt = 0.001
        temp_end_x,temp_end_y,temp_L0,temp_theta0 = self.Forward_Kinematics(self.theta1+self.theta1_dot*dt,self.theta2+self.theta2_dot*dt)
        self.end_x_dot = (temp_end_x - self.end_x) / dt
        self.end_y_dot = (temp_end_y - self.end_y) / dt
        self.L0_dot = (temp_L0 - self.L0) / dt
        self.theta0_dot = (temp_theta0 - self.theta0) / dt
        self.alpha = self.theta0 - np.pi / 2 * np.ones(self.alpha.shape)
        self.alpha_dot = self.theta0_dot


    def Forward_Kinematics(self, theta1, theta2):
        end_x = self.b * np.ones(theta1.shape) + self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1+theta2)
        end_y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1+theta2)
        L0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_y, end_x)
        return end_x, end_y, L0, theta0

    def VMC(self, F, T):
        T1 = np.zeros((self.num_envs,2), dtype=float)
        T2 = np.zeros((self.num_envs,2), dtype=float)
        t11 = self.l1*np.sin(self.theta0-self.theta1)-self.l2*np.sin(self.theta1+self.theta2-self.theta0)

        t12 = self.l1*np.cos(self.theta0-self.theta1)-self.l2*np.cos(self.theta1+self.theta2-self.theta0)
        t12 = t12/self.L0

        t21 = -self.l2*np.sin(self.theta1+self.theta2-self.theta0)

        t22 = -self.l2*np.cos(self.theta1+self.theta2-self.theta0)
        t22 = t22/self.L0
        
        T1 = t11*F + t12*T
        T2 = t21*F + t22*T

        return T1, T2

    def PD_Update(self, L0_reference, alpha_reference):
        F = self.L0_kp*(L0_reference-self.L0) + self.L0_kd*(0-self.L0_dot) + self.L0_ff
        T = self.alpha_kp*(alpha_reference-self.alpha) + self.alpha_kd*(0-self.alpha_dot)
        return self.VMC(F,-T)

