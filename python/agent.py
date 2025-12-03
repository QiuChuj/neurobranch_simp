import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class REINFORCE_Agent:
    def __init__(self, model, learning_rate=1e-4, gamma=0.99):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma

        # 存储轨迹
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        在训练阶段调用，根据概率选择动作
        """
        # 1. 确保状态是 Tensor 且有 Batch 维度
        state = torch.FloatTensor(state).unsqueeze(0)

        # 2. 网络输出 Logits (未归一化的分数)
        scores = self.model(state)

        # 3. 转换为概率
        probs = torch.softmax(scores, dim=1)

        # 4. 构建分布并采样
        # Categorical 自动处理多项分布采样
        m = Categorical(probs)
        action = m.sample()

        # 5. 关键：保存 log_prob 用于后续求导
        self.saved_log_probs.append(m.log_prob(action))

        return action.item()

    def update(self):
        """
        一局结束后调用，更新网络
        """
        R = 0
        policy_loss = []
        returns = []

        # 1. 计算折扣回报 (从后往前)
        # 每一以后的奖励都要打折加到当前步
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # 2. 归一化回报 (Baseline Trick)
        # 这一步极其重要！如果不做，训练很难收敛。
        # 它的作用是告诉模型：“比平均分高的动作是好动作，比平均分低的是坏动作”
        eps = 1e-9
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # 3. 计算策略梯度 Loss
        # Loss = - log(prob) * return
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # 4. 反向传播
        self.optimizer.zero_grad()
        # 求和所有步的 loss
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 5. 清空缓存，准备下一局
        del self.saved_log_probs[:]
        del self.rewards[:]

        return loss.item()
