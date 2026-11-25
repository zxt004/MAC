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
import seaborn as sns  # 添加seaborn用于热图

# 设置 matplotlib 字体和样式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 25
plt.rcParams['axes.unicode_minus'] = False

# 添加模式选择
MODE = "BPI"  # 可以选择 "PLOT" 或 "BPI"


def meta():
    # 初始化环境
    env = env1.AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=2)

    # 生成训练和测试数据
    np.random.seed(30)
    meta_train_data = np.random.uniform(-50.0, 50.0, 10)
    np.random.seed(40)
    test_data = np.random.uniform(-50.0, 50.0, 100)

    # 训练和测试 Meta Actor-Critic 模型
    start_time1 = time.time()
    agent_meta = MetaActorCriticAgent(env)
    agent_meta.maml_train(meta_train_data)
    # agent_meta.save_models('../model/meta')
    # agent_meta.load_models('../model/meta')
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
        print(f"\n=== 调试 {agent_name} 的返回格式 ===")
        print(f"Type: {type(test_results)}")
        print(f"Length: {len(test_results) if hasattr(test_results, '__len__') else 'No length'}")

        if hasattr(test_results, '__len__') and len(test_results) > 0:
            print(f"First element type: {type(test_results[0])}")
            print(f"First element: {test_results[0]}")
            if len(test_results) > 1:
                print(f"Second element type: {type(test_results[1])}")
                print(f"Second element: {test_results[1]}")

        if isinstance(test_results, tuple):
            # DQN 返回的是元组格式 (rewards_list, info_dict)
            print("Detected tuple format (likely DQN)")
            if len(test_results) >= 1 and isinstance(test_results[0], list):
                rewards = [float(x) for x in test_results[0]]
                print(f"Extracted rewards from tuple: {rewards[:5]}...")
                return rewards
            else:
                print("Invalid tuple format, returning zeros...")
                return [0] * 100
        elif isinstance(test_results, list) and len(test_results) > 0:
            if isinstance(test_results[0], dict):
                # 如果返回的是字典列表 (Meta AC的格式)
                rewards = [result['TotalReward'] for result in test_results]
                print(f"Extracted rewards (dict format): {rewards[:5]}...")  # 显示前5个
                return rewards
            elif isinstance(test_results[0], (int, float, np.number)):
                # 如果返回的是数值列表
                rewards = [float(x) for x in test_results]
                print(f"Direct numeric format: {rewards[:5]}...")  # 显示前5个
                return rewards
            elif isinstance(test_results[0], (list, tuple, np.ndarray)):
                # 如果返回的是嵌套列表/数组，尝试展平
                print("Detected nested structure, attempting to flatten...")
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
                print(f"Flattened rewards: {flattened[:5]}...")
                return flattened
            else:
                print(f"Unknown format, converting to list...")
                try:
                    return [float(x) for x in test_results]
                except:
                    return [0.0] * len(test_results)
        else:
            print("Empty or invalid results, returning zeros...")
            return [0.0] * 100  # 默认返回100个0

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

    # 打印结果
    print("rewards_MAC.mean: ", meta_mean)
    print("time_MAC: ", meta_time)
    print("rewards_AC.mean: ", no_meta_mean)
    print("time_AC: ", no_meta_time)
    print("rewards_DQN.mean: ", dqn_mean)
    print("time_DQN: ", dqn_time)
    print("rewards_PG.mean: ", pg_mean)
    print("time_PG: ", pg_time)

    # 根据模式选择执行不同的操作
    if MODE == "PLOT":
        # 绘制折线图模式
        # 确保奖励是一维数组
        rewards_meta = np.array(rewards_meta).flatten()
        rewards_no_meta = np.array(rewards_no_meta).flatten()
        rewards_dqn = np.array(rewards_dqn).flatten()
        rewards_pg = np.array(rewards_pg).flatten()

        # 计算差异
        dif1 = rewards_meta - rewards_no_meta
        dif2 = rewards_meta - rewards_dqn
        dif3 = rewards_meta - rewards_pg

        # 创建一行三列的图表布局
        fig = plt.figure(figsize=(32, 8))
        gs = gridspec.GridSpec(1, 3)

        # MAC-AC差异图
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(dif1, marker='.', linestyle='-', color='g', markersize=8)
        ax1.set_title('MAC - AC')
        ax1.set_xlabel('Test Data Index')
        ax1.set_ylabel('Reward Difference')
        ax1.grid(True)

        # MAC-DQN差异图
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(dif2, marker='.', linestyle='-', color='y', markersize=8)
        ax2.set_title('MAC - DQN')
        ax2.set_xlabel('Test Data Index')
        ax2.set_ylabel('Reward Difference')
        ax2.grid(True)

        # MAC-PG差异图
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(dif3, marker='.', linestyle='-', color='purple', markersize=8)
        ax3.set_title('MAC - PG')
        ax3.set_xlabel('Test Data Index')
        ax3.set_ylabel('Reward Difference')
        ax3.grid(True)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 1])

        # 保存图像为SVG格式
        plt.savefig('results_comparison_with_pg.svg', format='svg')
        plt.show()

        # 保存结果到.pt文件
        results = {
            'rewards_meta': rewards_meta,
            'rewards_no_meta': rewards_no_meta,
            'rewards_dqn': rewards_dqn,
            'rewards_pg': rewards_pg,
            'time_meta': meta_time,
            'time_no_meta': no_meta_time,
            'time_dqn': dqn_time,
            'time_pg': pg_time,
            'mean_meta': meta_mean,
            'mean_no_meta': no_meta_mean,
            'mean_dqn': dqn_mean,
            'mean_pg': pg_mean,
            'dif_meta_ac': dif1,
            'dif_meta_dqn': dif2,
            'dif_meta_pg': dif3
        }

        torch.save(results, 'experiment_results_with_pg.pt')
        print("Results saved to experiment_results_with_pg.pt")


    elif MODE == "BPI":

        # 计算 BPI 模式

        # 计算 rer 和 tir 函数

        def rer(mean):

            return ref_reward / mean

        def tir(time):

            return (max_time - time) / max_time

        # 计算 BPI 百分比

        def calculate_bpi(mean, time):

            r = rer(mean)

            t = tir(time)

            bpi = 0.8 * r + 0.2 * t

            return bpi * 100

        # 确定参考奖励和最大时间

        # 这里需要根据实际情况设置 ref_reward 和 max_time

        # 假设 ref_reward 是所有算法中最大的平均奖励

        ref_reward = max(meta_mean, no_meta_mean, dqn_mean, pg_mean)

        # 假设 max_time 是所有算法中最长的训练时间

        max_time = max(meta_time, no_meta_time, dqn_time, pg_time)

        # 计算每个算法的 BPI

        bpi_MAC = calculate_bpi(meta_mean, meta_time)

        bpi_AC = calculate_bpi(no_meta_mean, no_meta_time)

        bpi_DQN = calculate_bpi(dqn_mean, dqn_time)

        bpi_PG = calculate_bpi(pg_mean, pg_time)

        print(f"\nBPI Results:")

        print(f"MAC BPI: {bpi_MAC:.3f}")

        print(f"AC BPI: {bpi_AC:.3f}")

        print(f"DQN BPI: {bpi_DQN:.3f}")

        print(f"PG BPI: {bpi_PG:.3f}")

        # 创建更美观的柱状图显示BPI结果

        plt.figure(figsize=(12, 8))

        # 设置颜色方案

        colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']

        algorithms = ['MAC', 'AC', 'DQN', 'PG']

        bpi_values = [bpi_MAC, bpi_AC, bpi_DQN, bpi_PG]

        # 创建柱状图

        bars = plt.bar(algorithms, bpi_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        # 设置标题和标签

        plt.ylabel('BPI Score', fontsize=20, fontweight='bold')

        plt.title('BPI Comparison of Algorithms', fontsize=22, fontweight='bold', pad=20)

        # 设置y轴范围，使图表看起来更专业

        y_min = min(bpi_values) * 0.9

        y_max = max(bpi_values) * 1.1

        plt.ylim(y_min, y_max)

        # 添加网格线

        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在柱子上添加数值标签

        for bar, value in zip(bars, bpi_values):
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

        # 保存为高质量图像

        plt.savefig('bpi_comparison.svg', format='svg', dpi=300, bbox_inches='tight')

        plt.savefig('bpi_comparison.png', format='png', dpi=300, bbox_inches='tight')

        plt.show()

        # 创建详细的结果表格

        # print("\n" + "=" * 60)
        #
        # print("DETAILED PERFORMANCE COMPARISON")
        #
        # print("=" * 60)
        #
        # print(f"{'Algorithm':<10} {'Mean Reward':<15} {'Time (s)':<12} {'BPI Score':<10}")
        #
        # print("-" * 60)
        #
        # print(f"{'MAC':<10} {meta_mean:<15.3f} {meta_time:<12.3f} {bpi_MAC:<10.3f}")
        #
        # print(f"{'AC':<10} {no_meta_mean:<15.3f} {no_meta_time:<12.3f} {bpi_AC:<10.3f}")
        #
        # print(f"{'DQN':<10} {dqn_mean:<15.3f} {dqn_time:<12.3f} {bpi_DQN:<10.3f}")
        #
        # print(f"{'PG':<10} {pg_mean:<15.3f} {pg_time:<12.3f} {bpi_PG:<10.3f}")
        #
        # print("=" * 60)


if __name__ == '__main__':
    meta()