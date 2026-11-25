import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from paper.env1 import AnomalyMetricEnv  # 使用文档1的环境
from ac_net import Actor, Critic


class MetaActorCriticAgent:
    def __init__(self, env, lr=0.01, scheduler_step_size=200, scheduler_gamma=0.5):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)
        self.lr = lr

        # 优化器和学习率调度器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_scheduler = StepLR(self.actor_optimizer,
                                      step_size=scheduler_step_size,
                                      gamma=scheduler_gamma)
        self.critic_scheduler = StepLR(self.critic_optimizer,
                                       step_size=scheduler_step_size,
                                       gamma=scheduler_gamma)

    def maml_train(self, val, inner_lr=0.01, gamma=0.98, batch=5):
        num_episodes = len(val)
        outer_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

        episode = 0
        while episode < num_episodes:
            task_losses = []  # 存储每个任务的损失
            num_tasks = val[episode: episode + batch]  # 当前批次任务

            for task in num_tasks:
                # 保存原始参数
                actor_original = [param.clone() for param in self.actor.parameters()]
                critic_original = [param.clone() for param in self.critic.parameters()]

                # 内层训练 - 使用当前参数
                inner_loss = self.run_episode(task, gamma)

                # 计算内层梯度
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                inner_loss.backward()

                # 手动更新参数（内层更新）
                with torch.no_grad():
                    for param, grad in zip(self.actor.parameters(), [p.grad for p in self.actor.parameters()]):
                        param.data -= inner_lr * grad
                    for param, grad in zip(self.critic.parameters(), [p.grad for p in self.critic.parameters()]):
                        param.data -= inner_lr * grad

                # 使用更新后的参数运行测试，计算元损失
                state = self.env.reset(task)
                meta_loss = 0
                done = False
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)

                    # 将状态转换为张量
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    reward_tensor = torch.tensor([reward], dtype=torch.float32)

                    # 计算价值和目标价值
                    value = self.critic(state_tensor)
                    next_value = self.critic(next_state_tensor) if not done else torch.tensor([0.0])
                    target = reward_tensor + gamma * next_value

                    # 计算损失
                    critic_loss = (target.detach() - value).pow(2).mean()

                    log_prob = torch.log(self.actor(state_tensor)[action])
                    advantage = target.detach() - value.detach()
                    actor_loss = -log_prob * advantage

                    meta_loss += actor_loss + critic_loss

                    state = next_state

                task_losses.append(meta_loss)

                # 恢复原始参数
                with torch.no_grad():
                    for param, orig in zip(self.actor.parameters(), actor_original):
                        param.data = orig.data
                    for param, orig in zip(self.critic.parameters(), critic_original):
                        param.data = orig.data

            # 外层更新 - 使用元损失
            outer_optimizer.zero_grad()
            meta_loss = torch.stack(task_losses).mean()
            meta_loss.backward()
            outer_optimizer.step()

            # 更新学习率
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            episode += batch

    def run_episode(self, task, gamma):
        """运行一个完整的episode并返回损失"""
        state = self.env.reset(task)
        total_loss = 0
        done = False

        while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # 将状态转换为张量
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            # 计算价值和目标价值
            value = self.critic(state_tensor)
            next_value = self.critic(next_state_tensor) if not done else torch.tensor([0.0])
            target = reward_tensor + gamma * next_value

            # 计算Critic损失
            critic_loss = (target.detach() - value).pow(2).mean()

            # 计算Actor损失
            log_prob = torch.log(self.actor(state_tensor)[action])
            advantage = target.detach() - value.detach()
            actor_loss = -log_prob * advantage

            total_loss += actor_loss + critic_loss

            state = next_state

        return total_loss

    def test(self, val):
        num_episodes = len(val)
        self.actor.eval()
        self.critic.eval()

        results = []

        for episode in range(num_episodes):
            state = self.env.reset(val[episode])
            episode_reward = 0
            strategy_sequence = []
            state_sequence = [state.item()]
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # 记录策略名称
                if self.env.selected_strategies:
                    strategy_name = self.env.selected_strategies[-1]
                    strategy_sequence.append(strategy_name)

                episode_reward += reward
                state_sequence.append(next_state.item())
                state = next_state

            # 收集结果
            results.append({
                "InitialValue": val[episode],
                "FinalValue": state_sequence[-1],
                "Strategies": strategy_sequence,
                "TotalReward": episode_reward,
                "StateSequence": state_sequence
            })

        return results

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        action = probs.argmax().item()  # 选择概率最高的动作
        return action

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))


if __name__ == '__main__':
    # 创建环境实例
    env = AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)
    agent = MetaActorCriticAgent(env)

    # 生成训练数据 (100个随机初始值)
    train_data = np.random.uniform(-40, 40, 100)
    test_data = np.random.uniform(-40, 40, 20)

    # 训练和测试
    agent.maml_train(train_data)
    test_results = agent.test(test_data)

    # 打印测试结果
    for i, res in enumerate(test_results):
        print(f"Test {i + 1}:")
        print(f"  Initial: {res['InitialValue']:.2f}, Final: {res['FinalValue']:.2f}")
        print(f"  Strategies: {res['Strategies']}")
        print(f"  Reward: {res['TotalReward']:.2f}")
        print("-" * 50)