# NeuroBranch_simp
```
葛老师的建议：不仅要考虑变量方面，也要考虑子句方面，跟据子句的重要性来决定变量的重要性
```
这是neurobranch的简化版，使用一个较为简单的回归模型来计算分数。需要考虑的参数有：
### 1、变量方面
- 每个变量在所有子句中出现的次数
- 每个变量的决策层
- 每个变量的决策顺序（你是第几个被赋值的）
- 每个变量在冲突分析中出现的次数
- 每个变量在生成子句中出现的次数
- 每个变量的极性分布（正文字和负文字的占比）
- 每个变量在短子句中出现的次数（一个或两个变量构成的子句）
### 2、子句方面，如何衡量一个子句的重要性呢？
- LBD（LBD=2的子句称为glue clause）
- 子句长度（越短的子句越有决定性）

最终输入的参数：
- 每个变量在所有子句中出现的次数
- 每个变量的决策层
- 每个变量的决策顺序（你是第几个被赋值的）
- 每个变量在冲突分析中出现的次数
- 每个变量在生成子句中出现的次数
- 每个变量的极性分布（正文字和负文字的占比）
- 每个变量在短子句中出现的次数（一个或两个变量构成的子句）
- 每个变量所在子句的最小LBD

### 3、强化学习改进
- 在共享内存中添加输出奖励，作为强化学习的reward
- 在kissat中，执行完decision之后，使用公式计算reward，并输出到共享内存中
- 强化学习根据奖励来计算loss，然后更新网络

# REINFORCE 算法详解

### 1. 什么是 REINFORCE？

REINFORCE 是最早期的策略梯度（Policy Gradient）算法之一。
与 Q-Learning（DQN）不同，REINFORCE 不去计算每个动作的“价值”（Value），而是直接优化策略网络（Policy Network）。

- 输入：状态 $s$（你的 SAT 问题特征）。

- 输出：策略 $\pi(a|s)$（一个概率分布，表示选每个变量的概率）。

- 目标：调整网络参数 $\theta$，使得总回报（Reward）最大化。

它被称为 蒙特卡洛（Monte Carlo） 方法，意味着它必须完整地玩完一局游戏（Episode），拿到最终的胜负结果后，才能回头更新网络。

### 2. 核心直觉：它是如何学习的？

想象你在训练一只狗狗（Agent）：

- 尝试：你扔出飞盘，狗狗随机做了一系列动作：跑 -> 跳 -> 咬住 -> 跑回来。

- 结果：狗狗成功带回飞盘。

- 奖励：你给它一块肉干（Reward = +1）。

- 强化：REINFORCE 算法会说：“刚才那一串动作（跑、跳、咬）都是好的！增加下次遇到类似情况做这些动作的概率。”

关键点：
如果狗狗没接住（Reward = -1），算法会说：“这一串动作里肯定有问题的，降低这一串动作在未来出现的概率。”

虽然我们不知道具体是“跳”得不好，还是“咬”得不好，但只要尝试次数足够多，统计学规律会让那些总是出现在成功序列里的动作脱颖而出。

### 3. 数学原理（简化版）

我们的目标是最大化期望回报 $J(\theta)$。
根据 策略梯度定理（Policy Gradient Theorem），我们需要计算梯度的方向，然后更新网络参数：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{t=0}^{T} \nabla_\theta \log \pi(a_t | s_t) \cdot G_t$$

- $\pi(a_t | s_t)$：网络在状态 $s_t$ 选择动作 $a_t$ 的概率。

- $\log \pi$：取对数是因为数学推导方便，且数值更稳定。

- $G_t$：从时间步 $t$ 开始，一直到游戏结束的累计折现回报（Discounted Return）。

- $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$

- $\gamma$（Gamma）是折扣因子，越接近 1 代表越看重长远利益。

公式翻译成人话：
更新幅度 = (这个动作的对数概率梯度) $\times$ (这个动作最终带来了多少好处 $G_t$)

如果 $G_t$ 是正的大数 $\rightarrow$ 大幅增加该动作概率。

如果 $G_t$ 是负数 $\rightarrow$ 降低该动作概率。

### 4. 算法流程步骤

针对你的 SAT 求解器，流程如下：

初始化：建立你的神经网络 neurobranch_simp。

- 采样（Rollout）：把当前的 CNF 问题输入网络。网络输出概率，根据概率随机采样一个变量作为决策。求解器执行决策，环境更新，直到 SAT 或 UNSAT（一局结束）。

- 记录：保存这一局所有的 (状态, 动作, 奖励) 轨迹。

- 计算回报：从最后一刻往前推，计算每一步的 $G_t$。通常会做标准化（减均值除方差），这能极大地稳定训练。

- 反向传播：计算 Loss：$Loss = - \sum (\log \pi(a_t|s_t) \times G_t)$注意：因为 PyTorch 是梯度下降，我们要最大化回报，所以加负号最小化 Loss。执行 optimizer.step() 更新权重。

- 重复：开始下一局。

5. 为什么它适合你的 SAT 模型？

对于 SAT 求解问题，REINFORCE 比 DQN 更合适，原因如下：

- 处理庞大的动作空间：DQN 需要计算每个动作的 Q 值。如果你的变量有 1000 个甚至更多，Q-Table 或 Q-Network 很难收敛。REINFORCE 直接输出概率分布，更适合大规模离散动作。

- 随机性探索：SAT 求解非常容易陷入局部最优（Local Optima）。REINFORCE 是基于概率采样的（Stochastic Policy），这意味着即使网络认为变量 A 最好，它也有 5% 的概率选变量 B。这种天然的随机性对跳出死循环至关重要。

- 关注最终结果：SAT 是典型的“过程不重要，结果才重要”的问题。中间的 BCP 即使再多，如果最后解不出，也是白搭。REINFORCE 基于整局游戏的 $G_t$ 更新，能捕捉到长期的因果关系。

6. PyTorch 代码实现模板

这是专门适配你 neurobranch_simp 的训练核心代码：
```
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
```

### 7. REINFORCE 的优缺点

优点

- 简单：实现起来代码很少，逻辑清晰。

- 通用：几乎可以用于任何强化学习环境。

- 收敛性：理论上保证能收敛到局部最优。

缺点（以及如何解决）

- 方差大（High Variance）：因为是基于随机采样的，有时候运气好奖励高，有时候运气差奖励低，导致梯度乱跳。解决：使用 Reward Normalization（代码里已包含）或者引入 Critic 网络（变成 PPO/Actor-Critic 算法）。

- 样本效率低：必须玩完完整的一局才能更新一次。解决：多线程并行采样（一次跑 10 个求解器）。

### 总结

对于你的项目，REINFORCE 是最佳的入门级 RL 算法。它足够简单，能让你快速跑通流程；同时它直接优化策略的特性非常契合 SAT 这种“在一堆变量中选一个”的分类任务本质。