import gymnasium as gym
import json
from pathlib import Path
import argparse
import numpy as np
import time
from gymnasium.wrappers import RecordVideo
import os
import sys
import torch

# 添加scripts目录到Python路径以导入自定义agent
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# 尝试导入CollisionAwareBFTQ，如果不存在则跳过
try:
    from collision_aware_bftq import CollisionAwareBFTQ
    COLLISION_AWARE_AVAILABLE = True
except ImportError:
    COLLISION_AWARE_AVAILABLE = False
    print("[WARNING] CollisionAwareBFTQ不可用，将使用标准agent")

def fix_agent_device_compatibility(agent):
    """
    临时修复agent的设备兼容性问题，而不修改核心库
    """
    if hasattr(agent, 'bftq') and hasattr(agent.bftq, 'device'):
        device = agent.bftq.device
        # 如果device是torch.device对象，转换为字符串
        if hasattr(device, 'type') and hasattr(device, 'index'):
            if device.type == 'cuda' and device.index is not None:
                agent.bftq.device = f"cuda:{device.index}"
            else:
                agent.bftq.device = str(device)
        elif isinstance(device, str) and device.isdigit():
            # 如果是纯数字字符串，添加cuda前缀
            agent.bftq.device = f"cuda:{device}"
    
    return agent

def safe_load_model(agent, checkpoint_path):
    """
    安全加载模型，处理设备兼容性问题
    """
    try:
        # 方法1：直接尝试加载
        return agent.load(checkpoint_path)
    except (TypeError, RuntimeError) as e:
        print(f"[INFO] 第一次加载失败，尝试设备兼容性修复...")
        
        # 方法2：强制使用CPU进行可视化
        print(f"[INFO] 强制使用CPU模式以避免设备不兼容问题...")
        
        if hasattr(agent, 'bftq'):
            # 保存原始设备设置
            original_device = agent.bftq.device
            
            # 设置为CPU模式
            agent.bftq.device = 'cpu'
            
            # 如果有exploration_policy，也设置其设备
            if hasattr(agent, 'exploration_policy') and hasattr(agent.exploration_policy, 'pi_greedy'):
                if hasattr(agent.exploration_policy.pi_greedy, 'device'):
                    agent.exploration_policy.pi_greedy.device = 'cpu'
            
            try:
                result = agent.load(checkpoint_path)
                
                # 确保所有相关组件都在CPU上
                if hasattr(agent.bftq, '_value_network') and agent.bftq._value_network is not None:
                    agent.bftq._value_network = agent.bftq._value_network.cpu()
                
                print(f"[INFO] 成功使用CPU模式加载模型")
                return result
                
            except Exception as e3:
                # 如果还是失败，恢复原始设置并抛出异常
                agent.bftq.device = original_device
                raise e3
        else:
            raise e

def run_episode(env, agent, episode_id, unlimited_time=False):
    """
    运行一个回合并渲染环境，记录碰撞信息
    """
    done = False
    obs, _ = env.reset()
    total_reward = 0
    episode_steps = 0
    collision_occurred = False
    
    print(f"\n=== Episode {episode_id + 1} 开始 ===")
    
    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        episode_steps += 1
        
        # 检查是否发生碰撞
        if 'crashed' in info and info['crashed']:
            collision_occurred = True
            print(f"  💥 步骤 {episode_steps}: 发生碰撞!")
        
        env.render()  # 渲染环境
        time.sleep(0.05)  # 减少延迟，加快可视化速度
        
        if done or truncated:
            break
    
    # 打印episode统计
    status = "碰撞" if collision_occurred else "安全完成"
    print(f"=== Episode {episode_id + 1} 结束: {status}, 步数: {episode_steps}, 奖励: {total_reward:.2f} ===")
    
    return total_reward, episode_steps, collision_occurred

def load_collision_aware_agent(agent_config_path, env):
    """
    加载CollisionAwareBFTQ agent
    """
    with open(agent_config_path, 'r') as f:
        agent_config = json.load(f)
    
    # 设置为使用CollisionAwareBFTQ
    agent_config["__class__"] = "<class 'scripts.collision_aware_bftq.CollisionAwareBFTQ'>"
    
    # 确保必要的参数存在
    agent_config.setdefault("collision_penalty", 5.0)
    agent_config.setdefault("step_reward_weight", 0.05)
    agent_config.setdefault("log_interval", 200)
    agent_config.setdefault("enable_cvar", False)
    agent_config.setdefault("cvar_alpha", 0.1)
    
    # 创建agent
    agent = CollisionAwareBFTQ(env, agent_config)
    return agent

def visualize_agent(env_config, agent_config, checkpoint_path, num_episodes=10, video_dir=None, use_collision_aware=False, unlimited_time=False):
    """
    加载预训练的智能体并进行可视化
    """
    print(f"[INFO] 开始可视化智能体")
    print(f"[INFO] 检查点路径: {checkpoint_path}")
    print(f"[INFO] 使用CollisionAware模式: {use_collision_aware}")
    if unlimited_time:
        print(f"[INFO] 🕒 无限时间模式：移除时间限制以测试真正驾驶能力")
    
    # 加载环境
    env = load_environment(env_config)
    
    # 🔥 新功能：可选移除时间限制
    if unlimited_time:
        # 检查环境是否有TimeLimit wrapper
        if hasattr(env, '_max_episode_steps'):
            print(f"[INFO] 原始时间限制: {env._max_episode_steps}步")
            env._max_episode_steps = 10000  # 设置很大的值
            print(f"[INFO] 时间限制已设置为: {env._max_episode_steps}步")
        
        # 如果是TwoWayEnv，直接修改其config
        if hasattr(env.unwrapped, 'config') and 'duration' in env.unwrapped.config:
            original_duration = env.unwrapped.config['duration']
            env.unwrapped.config['duration'] = 10000
            print(f"[INFO] TwoWayEnv duration: {original_duration} → {env.unwrapped.config['duration']}")
    
    # 设置环境的渲染参数
    if hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure({
            'render_mode': 'rgb_array',  # 使用rgb_array模式以支持录制
            'offscreen_rendering': False
        })
    
    # 设置视频保存目录
    if video_dir is None:
        agent_type = "collision_aware" if use_collision_aware else "standard"
        time_mode = "_unlimited" if unlimited_time else ""
        video_dir = os.path.join('out', 'videos', 
                                f"{agent_type}_agent{time_mode}_{time.strftime('%Y%m%d-%H%M%S')}")
    
    # 包装环境以支持视频录制
    env = RecordVideo(env, 
                     video_folder=video_dir,
                     episode_trigger=lambda x: True)  # 录制所有回合
    
    # 加载agent
    if use_collision_aware and COLLISION_AWARE_AVAILABLE:
        print("[INFO] 使用CollisionAwareBFTQ agent")
        agent = load_collision_aware_agent(agent_config, env)
    else:
        if use_collision_aware and not COLLISION_AWARE_AVAILABLE:
            print("[WARNING] CollisionAware模式不可用，使用标准agent")
        print("[INFO] 使用标准BFTQAgent")
        agent = load_agent(agent_config, env)
    
    # 加载模型（使用安全加载方法）
    try:
        result = safe_load_model(agent, checkpoint_path)
        print(f"[INFO] ✅ 成功加载模型")
        
        # 🔥 关键修复：设置为评估模式
        agent.eval()
        print(f"[INFO] 🎯 已设置为评估模式（关闭探索，使用最优策略）")
        
        print(f"[INFO] 🎬 视频将保存在: {video_dir}")
    except Exception as e:
        print(f"❌ 加载模型时出现错误: {e}")
        print(f"提示：请检查模型文件是否损坏或不兼容")
        return

    # 统计信息
    total_rewards = []
    total_steps = []
    collision_count = 0
    timeout_count = 0  # 新增：超时计数
    
    print(f"\n🚗 开始运行 {num_episodes} 个episodes...")
    
    # 运行指定回合数
    try:
        for episode in range(num_episodes):
            reward, steps, collision = run_episode(env, agent, episode, unlimited_time)
            total_rewards.append(reward)
            total_steps.append(steps)
            if collision:
                collision_count += 1
            elif unlimited_time and steps >= 1000:  # 如果无限时间模式下步数很长
                timeout_count += 1
                
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
    finally:
        env.close()
        
        # 打印总体统计
        print(f"\n" + "="*60)
        print(f"🎯 总体表现统计 ({num_episodes} episodes)")
        print(f"="*60)
        
        if total_rewards and total_steps:
            print(f"碰撞次数: {collision_count}/{num_episodes}")
            print(f"碰撞率: {collision_count/num_episodes*100:.1f}%")
            if unlimited_time and timeout_count > 0:
                print(f"长距离驾驶次数: {timeout_count}/{num_episodes} (≥1000步)")
            print(f"平均奖励: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
            print(f"平均步数: {np.mean(total_steps):.1f} (±{np.std(total_steps):.1f})")
            print(f"最长episode: {max(total_steps)} 步")
            print(f"最短episode: {min(total_steps)} 步")
        else:
            print(f"❌ 没有成功完成任何episode")
            print(f"运行失败：可能是设备兼容性或模型问题")
            
        print(f"\n🎬 视频已保存在目录: {video_dir}")
        print(f"="*60)

def main():
    parser = argparse.ArgumentParser(description='可视化智能体轨迹')
    parser.add_argument('--env', type=str, required=True, help='环境配置文件路径')
    parser.add_argument('--agent', type=str, required=True, help='智能体配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--episodes', type=int, default=10, help='要运行的回合数')
    parser.add_argument('--video-dir', type=str, help='视频保存目录（可选）')
    parser.add_argument('--use-collision-aware', action='store_true', 
                       help='使用CollisionAwareBFTQ agent（仅适用于用--with-metrics训练的模型）')
    parser.add_argument('--unlimited-time', action='store_true',
                       help='移除时间限制，测试agent的真正驾驶能力')
    
    args = parser.parse_args()
    
    # 验证文件路径
    if not os.path.exists(args.env):
        print(f"❌ 环境配置文件不存在: {args.env}")
        return
    if not os.path.exists(args.agent):
        print(f"❌ 智能体配置文件不存在: {args.agent}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    visualize_agent(args.env, args.agent, args.checkpoint, args.episodes, args.video_dir, args.use_collision_aware, args.unlimited_time)

if __name__ == "__main__":
    main() 