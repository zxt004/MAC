from ac_no_meta import *
from ac_meta_paper import *
from PG import *
from DQN import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import torch
from paper import env1
import seaborn as sns
import random
import pandas as pd

# 设置 matplotlib 字体和样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.unicode_minus'] = False

# 添加模式选择
MODE = "BPI"  # 可以选择 "PLOT" 或 "BPI"

# 设置随机种子数量
NUM_SEEDS = 10


def meta(test_seed=None):
    # 初始化环境
    env = env1.AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)

    # 生成训练和测试数据
    np.random.seed(31)  # 固定训练数据种子
    meta_train_data = np.random.uniform(-50.0, 50.0, 40)

    # 使用传入的测试种子或默认值
    if test_seed is not None:
        np.random.seed(test_seed)
    else:
        np.random.seed(41)  # 默认测试种子

    test_data = np.random.uniform(-50.0, 50.0, 100)

    # 训练和测试 Meta Actor-Critic 模型
    start_time1 = time.time()
    agent_meta = MetaActorCriticAgent(env)
    # agent_meta.maml_train(meta_train_data)
    # agent_meta.save_models('../model/meta')
    agent_meta.load_models('../model/meta')
    test_results_meta = agent_meta.test(test_data)
    end_time1 = time.time()

    # 训练和测试 Actor-Critic 模型
    start_time2 = time.time()
    agent_no_meta = ActorCriticAgent(env)
    agent_no_meta.train(meta_train_data)
    agent_no_meta.save_models('../model/no_meta')
    agent_no_meta.load_models('../model/no_meta')
    test_results_no_meta = agent_no_meta.test(test_data)
    end_time2 = time.time()

    # 训练和测试 DQN 模型
    start_time3 = time.time()
    agent_dqn = DQNAgent(env)
    agent_dqn.train(meta_train_data)
    agent_dqn.save_models('../model/dqn')
    agent_dqn.load_models('../model/dqn')
    test_results_dqn = agent_dqn.test(test_data)
    end_time3 = time.time()

    # 训练和测试 Policy Gradient 模型
    start_time4 = time.time()
    agent_pg = PolicyGradientAgent(env)
    agent_pg.train(meta_train_data)
    agent_pg.save_models('../model/pg')
    agent_pg.load_models('../model/pg')
    test_results_pg = agent_pg.test(test_data)
    end_time4 = time.time()

    # 提取奖励值 - 处理不同的返回格式
    def extract_rewards(test_results, agent_name="Unknown"):
        """提取测试结果中的奖励值"""
        if isinstance(test_results, tuple):
            # DQN 返回的是元组格式 (rewards_list, info_dict)
            if len(test_results) >= 1 and isinstance(test_results[0], list):
                rewards = [float(x) for x in test_results[0]]
                return rewards
            else:
                return [0] * 100
        elif isinstance(test_results, list) and len(test_results) > 0:
            if isinstance(test_results[0], dict):
                # 如果返回的是字典列表 (Meta AC的格式)
                rewards = [result['TotalReward'] for result in test_results]
                return rewards
            elif isinstance(test_results[0], (int, float, np.number)):
                # 如果返回的是数值列表
                rewards = [float(x) for x in test_results]
                return rewards
            elif isinstance(test_results[0], (list, tuple, np.ndarray)):
                # 如果返回的是嵌套列表/数组，尝试展平
                flattened = []
                for item in test_results:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        if len(item) > 0:
                            # 假设最后一个元素是奖励值
                            if isinstance(item[-1], (int, float, np.number)):
                                flattened.append(float(item[-1]))
                            # 或者假设第一个元素是奖励值
                            elif isinstance(item[0], (int, float, np.number)):
                                flattened.append(float(item[0]))
                            else:
                                # 尝试求和作为总奖励
                                try:
                                    total_reward = sum(
                                        [float(x) for x in item if isinstance(x, (int, float, np.number))])
                                    flattened.append(total_reward)
                                except:
                                    flattened.append(0.0)
                        else:
                            flattened.append(0.0)
                    elif isinstance(item, (int, float, np.number)):
                        flattened.append(float(item))
                    else:
                        flattened.append(0.0)
                return flattened
            else:
                try:
                    return [float(x) for x in test_results]
                except:
                    return [0.0] * len(test_results)
        else:
            return [0.0] * 100

    rewards_meta = extract_rewards(test_results_meta, "Meta AC")
    rewards_no_meta = extract_rewards(test_results_no_meta, "AC")
    rewards_dqn = extract_rewards(test_results_dqn, "DQN")
    rewards_pg = extract_rewards(test_results_pg, "PG")

    # 计算统计信息
    meta_mean = np.mean(rewards_meta)
    no_meta_mean = np.mean(rewards_no_meta)
    dqn_mean = np.mean(rewards_dqn)
    pg_mean = np.mean(rewards_pg)
    meta_time = end_time1 - start_time1
    no_meta_time = end_time2 - start_time2
    dqn_time = end_time3 - start_time3
    pg_time = end_time4 - start_time4

    # 计算 BPI
    def rer(mean, ref_reward):
        return ref_reward / mean

    def tir(time, max_time):
        return (max_time - time) / max_time

    def calculate_bpi(mean, time, ref_reward, max_time):
        r = rer(mean, ref_reward)
        t = tir(time, max_time)
        bpi = 0.8 * r + 0.2 * t
        return bpi * 100

    # 确定参考奖励和最大时间
    ref_reward = max(meta_mean, no_meta_mean, dqn_mean, pg_mean)
    max_time = max(meta_time, no_meta_time, dqn_time, pg_time)

    # 计算每个算法的 BPI
    bpi_MAC = calculate_bpi(meta_mean, meta_time, ref_reward, max_time)
    bpi_AC = calculate_bpi(no_meta_mean, no_meta_time, ref_reward, max_time)
    bpi_DQN = calculate_bpi(dqn_mean, dqn_time, ref_reward, max_time)
    bpi_PG = calculate_bpi(pg_mean, pg_time, ref_reward, max_time)

    return {
        'bpi_MAC': bpi_MAC,
        'bpi_AC': bpi_AC,
        'bpi_DQN': bpi_DQN,
        'bpi_PG': bpi_PG,
        'mean_MAC': meta_mean,
        'mean_AC': no_meta_mean,
        'mean_DQN': dqn_mean,
        'mean_PG': pg_mean,
        'time_MAC': meta_time,
        'time_AC': no_meta_time,
        'time_DQN': dqn_time,
        'time_PG': pg_time
    }


if __name__ == '__main__':
    # 随机选择10个不同的测试种子
    random_seeds = random.sample(range(1000), NUM_SEEDS)

    # 存储每个种子的BPI结果
    bpi_results = {
        'MAC': [],
        'AC': [],
        'DQN': [],
        'PG': []
    }

    # 存储平均值和时间
    mean_results = {
        'MAC': [],
        'AC': [],
        'DQN': [],
        'PG': []
    }

    time_results = {
        'MAC': [],
        'AC': [],
        'DQN': [],
        'PG': []
    }

    # 对每个种子运行实验
    for i, seed in enumerate(random_seeds):
        print(f"Running experiment {i + 1}/{NUM_SEEDS} with seed {seed}")
        results = meta(test_seed=seed)

        # 收集BPI结果
        bpi_results['MAC'].append(results['bpi_MAC'])
        bpi_results['AC'].append(results['bpi_AC'])
        bpi_results['DQN'].append(results['bpi_DQN'])
        bpi_results['PG'].append(results['bpi_PG'])

        # 收集平均值和时间
        mean_results['MAC'].append(results['mean_MAC'])
        mean_results['AC'].append(results['mean_AC'])
        mean_results['DQN'].append(results['mean_DQN'])
        mean_results['PG'].append(results['mean_PG'])

        time_results['MAC'].append(results['time_MAC'])
        time_results['AC'].append(results['time_AC'])
        time_results['DQN'].append(results['time_DQN'])
        time_results['PG'].append(results['time_PG'])

    # 计算平均BPI
    avg_bpi_MAC = np.mean(bpi_results['MAC'])
    avg_bpi_AC = np.mean(bpi_results['AC'])
    avg_bpi_DQN = np.mean(bpi_results['DQN'])
    avg_bpi_PG = np.mean(bpi_results['PG'])

    # 打印平均结果
    print("\n" + "=" * 60)
    print("AVERAGE RESULTS ACROSS ALL SEEDS")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'Avg BPI':<15} {'Avg Reward':<15} {'Avg Time (s)':<15}")
    print("-" * 60)
    print(
        f"{'MAC':<10} {avg_bpi_MAC:<15.3f} {np.mean(mean_results['MAC']):<15.3f} {np.mean(time_results['MAC']):<15.3f}")
    print(f"{'AC':<10} {avg_bpi_AC:<15.3f} {np.mean(mean_results['AC']):<15.3f} {np.mean(time_results['AC']):<15.3f}")
    print(
        f"{'DQN':<10} {avg_bpi_DQN:<15.3f} {np.mean(mean_results['DQN']):<15.3f} {np.mean(time_results['DQN']):<15.3f}")
    print(f"{'PG':<10} {avg_bpi_PG:<15.3f} {np.mean(mean_results['PG']):<15.3f} {np.mean(time_results['PG']):<15.3f}")
    print("=" * 60)

    # 创建热力图数据
    heatmap_data = np.array([bpi_results['MAC'],
                             bpi_results['AC'],
                             bpi_results['DQN'],
                             bpi_results['PG']]).T

    # 创建DataFrame以便更好地显示标签
    algorithms = ['MAC', 'AC', 'DQN', 'PG']
    seed_labels = [f'Seed {i + 1}\n({seed})' for i, seed in enumerate(random_seeds)]

    df = pd.DataFrame(heatmap_data, index=seed_labels, columns=algorithms)

    # 创建热力图
    plt.figure(figsize=(14, 10))

    # 使用seaborn绘制热力图
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd",
                     cbar_kws={'label': 'BPI Score'},
                     linewidths=0.5, linecolor='gray')

    # 设置标题和标签
    plt.title('BPI Scores Across Different Algorithms and Random Seeds',
              fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Algorithms', fontsize=20, fontweight='bold')
    plt.ylabel('Random Seeds', fontsize=20, fontweight='bold')

    # 调整颜色条标签大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('BPI Score', fontsize=18, fontweight='bold')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('bpi_heatmap.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig('bpi_heatmap.png', format='png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

    # 创建平均BPI的柱状图
    plt.figure(figsize=(10, 8))

    # 设置颜色
    colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']

    # 创建柱状图
    algorithms = ['MAC', 'AC', 'DQN', 'PG']
    avg_bpi_values = [avg_bpi_MAC, avg_bpi_AC, avg_bpi_DQN, avg_bpi_PG]

    bars = plt.bar(algorithms, avg_bpi_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # 设置标题和标签
    plt.title('Average BPI Scores Across All Seeds', fontsize=22, fontweight='bold', pad=20)
    plt.ylabel('Average BPI Score', fontsize=20, fontweight='bold')

    # 设置y轴范围
    y_min = min(avg_bpi_values) * 0.9
    y_max = max(avg_bpi_values) * 1.1
    plt.ylim(y_min, y_max)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上添加数值标签
    for bar, value in zip(bars, avg_bpi_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # 添加参考线
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Reference Line (100)')

    # 添加图例
    plt.legend(loc='upper right', fontsize=14)

    # 美化坐标轴
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('average_bpi_bar_chart.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig('average_bpi_bar_chart.png', format='png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

    # 保存详细结果到文件
    results_dict = {
        'seeds': random_seeds,
        'bpi_results': bpi_results,
        'mean_results': mean_results,
        'time_results': time_results,
        'avg_bpi': {
            'MAC': avg_bpi_MAC,
            'AC': avg_bpi_AC,
            'DQN': avg_bpi_DQN,
            'PG': avg_bpi_PG
        }
    }

    torch.save(results_dict, 'multiple_seeds_results.pt')
    print("Detailed results saved to multiple_seeds_results.pt")