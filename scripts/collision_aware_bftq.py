"""
基于碰撞率的BFTQ算法扩展
适用于highway环境的碰撞检测和损失函数动态调整
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import logging
import csv
from typing import Dict, Any, List, Tuple
from collections import deque

from rl_agents.agents.budgeted_ftq.agent import BFTQAgent

logger = logging.getLogger(__name__)

# 设置matplotlib日志级别为WARNING，减少调试信息
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class CollisionAwareBFTQ(BFTQAgent):
    """
    带有碰撞感知功能的BFTQ智能体
    
    功能：
    1. 跟踪碰撞率和episode步数
    2. 将碰撞转换为cost信号用于BFTQ训练
    3. 实时显示和保存碰撞统计数据
    
    碰撞率计算方式：
    - episode级碰撞率：collision_rate = 碰撞episode数 / 总episode数
    - 每个episode要么安全(0)，要么碰撞(1)
    - 将碰撞转换为步级cost(1/0)传递给BFTQ训练
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            # 碰撞感知配置
            "collision_penalty": 10.0,          # 碰撞惩罚
            "step_reward_weight": 0.1,          # 步数奖励权重
            "log_interval": 100,                # 日志输出间隔
            
            # CVaR风险管理配置
            "enable_cvar": False,               # 是否启用CVaR风险管理
            "cvar_alpha": 0.1,                  # CVaR置信水平(关注最差的10%)
            "cvar_window_size": 100,            # CVaR计算窗口大小
            "cvar_penalty_weight": 2.0,         # CVaR惩罚权重
            "cvar_min_episodes": 10,            # CVaR启用的最小episode数
        })
        return config

    def __init__(self, env, config=None):
        super().__init__(env, config)
        
        # 初始化碰撞指标跟踪
        self.reset_collision_metrics()
        
        # 初始化CSV保存
        self.csv_file = None
        self.csv_writer = None
        
        # 添加episode长度跟踪
        self.episode_lengths = []  # 记录每个episode的长度
        
        # CVaR相关参数
        self.enable_cvar = config.get('enable_cvar', False)  # 是否启用CVaR风险管理
        self.cvar_alpha = config.get('cvar_alpha', 0.1)  # CVaR置信水平(关注最差的10%)
        self.cvar_window_size = config.get('cvar_window_size', 100)  # CVaR计算窗口
        self.cvar_penalty_weight = config.get('cvar_penalty_weight', 2.0)  # CVaR惩罚权重
        self.cvar_min_episodes = config.get('cvar_min_episodes', 10)  # CVaR启用的最小episode数
        
        if self.enable_cvar:
            logger.info("CollisionAwareBFTQ: 碰撞感知BFTQ智能体已创建(含CVaR风险管理)")
        else:
            logger.info("CollisionAwareBFTQ: 碰撞感知BFTQ智能体已创建(不含CVaR)")

    def set_writer(self, writer):
        """重写set_writer方法，在writer设置后初始化CSV"""
        super().set_writer(writer)
        self.setup_csv_logging()
        logger.info(f"CollisionAwareBFTQ: Writer已设置，CSV文件初始化完成")

    def reset_collision_metrics(self):
        """重置碰撞指标跟踪"""
        # 当前episode指标
        self.episode_steps = 0
        self.episode_had_collision = False  # 标记当前episode是否发生过碰撞
        
        # 全局累计指标
        self.total_collision_episodes = 0  # 发生碰撞的episode总数
        self.total_episodes = 0
        self.step_count = 0
        
        # 重置episode长度记录
        self.episode_lengths = []
        
        logger.info("CollisionAwareBFTQ: 碰撞指标已重置")

    def setup_csv_logging(self):
        """设置CSV日志文件"""
        if self.writer is not None:
            # 如果CSV文件已经打开，先关闭它
            if self.csv_file is not None:
                self.csv_file.close()
                
            log_dir = str(self.writer.logdir)
            self.csv_path = os.path.join(log_dir, "collision_metrics.csv")
            self.csv_file = open(self.csv_path, 'w', newline='')
            fieldnames = ['episode', 'collision_rate', 'episode_length', 'had_collision']
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            logger.info(f"CollisionAwareBFTQ: CSV文件已创建: {self.csv_path}")
        else:
            logger.warning("CollisionAwareBFTQ: Writer为空，无法创建CSV文件")

    def extract_collision_info(self, info: dict) -> bool:
        """从info中提取碰撞信息"""
        if info is None:
            return False
        
        # 常见的碰撞信息字段名
        crash_keys = ['crashed', 'collision', 'crash', 'is_crashed', 'vehicle_crashed']
        
        for key in crash_keys:
            if key in info:
                return bool(info[key])
        
        # 如果没有找到直接的碰撞信息，尝试从其他字段推断
        if 'rewards' in info:
            rewards = info['rewards']
            if isinstance(rewards, (list, np.ndarray)) and len(rewards) > 0:
                # 如果奖励小于某个阈值，可能是碰撞
                min_reward = np.min(rewards)
                if min_reward < -5:  # 可根据环境调整阈值
                    return True
        
        return False
    
    def calculate_cvar_threshold(self) -> float:
        """
        计算episode长度的CVaR阈值
        
        :return: α分位数阈值，低于此值的episode被认为是worst case
        """
        if not self.enable_cvar:  # CVaR未启用
            return 0.0
            
        if len(self.episode_lengths) < self.cvar_min_episodes:  # 需要足够的历史数据
            return 0.0
            
        # 使用最近的窗口数据计算CVaR
        recent_lengths = self.episode_lengths[-self.cvar_window_size:]
        threshold = np.percentile(recent_lengths, self.cvar_alpha * 100)
        return threshold
    
    def calculate_cvar_penalty(self, current_length: int, is_done: bool) -> float:
        """
        基于CVaR计算penalty
        
        :param current_length: 当前episode长度
        :param is_done: 是否episode结束
        :return: CVaR penalty值
        """
        if not self.enable_cvar:  # CVaR未启用
            return 0.0
            
        if not is_done or len(self.episode_lengths) < self.cvar_min_episodes:
            return 0.0
            
        cvar_threshold = self.calculate_cvar_threshold()
        
        # 如果当前episode属于worst case (长度低于CVaR阈值)
        if current_length <= cvar_threshold:
            # 计算相对于CVaR阈值的差距
            penalty_ratio = (cvar_threshold - current_length) / max(cvar_threshold, 1.0)
            cvar_penalty = penalty_ratio * self.cvar_penalty_weight
            return cvar_penalty
        
        return 0.0
    
    def calculate_cvar_cost(self, current_length: int) -> float:
        """
        基于CVaR计算额外的cost
        
        :param current_length: 当前episode长度  
        :return: CVaR cost值
        """
        if not self.enable_cvar:  # CVaR未启用
            return 0.0
            
        if len(self.episode_lengths) < self.cvar_min_episodes:
            return 0.0
            
        cvar_threshold = self.calculate_cvar_threshold()
        
        # 如果当前episode正在朝worst case发展
        if current_length > 0:
            expected_final_length = current_length  # 保守估计
            if expected_final_length <= cvar_threshold * 1.2:  # 在风险区域内
                risk_factor = max(0, (cvar_threshold * 1.2 - expected_final_length) / max(cvar_threshold, 1.0))
                cvar_cost = risk_factor * 0.5  # 较小的渐进cost
                return cvar_cost
        
        return 0.0
    
    def get_cvar_statistics(self) -> Dict[str, float]:
        """
        获取CVaR相关统计信息
        
        :return: CVaR统计字典
        """
        if not self.enable_cvar:  # CVaR未启用
            return {'cvar_threshold': 0.0, 'cvar_value': 0.0, 'worst_case_episodes': 0, 'worst_case_rate': 0.0}
            
        if len(self.episode_lengths) < self.cvar_min_episodes:
            return {'cvar_threshold': 0.0, 'cvar_value': 0.0, 'worst_case_episodes': 0, 'worst_case_rate': 0.0}
            
        cvar_threshold = self.calculate_cvar_threshold()
        worst_case_episodes = sum(1 for length in self.episode_lengths if length <= cvar_threshold)
        worst_case_rate = worst_case_episodes / len(self.episode_lengths)
        
        # 计算CVaR值（worst case的平均长度）
        worst_case_lengths = [length for length in self.episode_lengths if length <= cvar_threshold]
        cvar_value = np.mean(worst_case_lengths) if worst_case_lengths else 0.0
        
        return {
            'cvar_threshold': cvar_threshold,
            'cvar_value': cvar_value,
            'worst_case_episodes': worst_case_episodes,
            'worst_case_rate': worst_case_rate
        }

    def record(self, state, action, reward, next_state, done, info):
        """记录transition并处理碰撞指标，鼓励更长的episode"""
        if not self.training:
            return super().record(state, action, reward, next_state, done, info)

        # 检测episode开始（第一步）
        if self.episode_steps == 0:
            logger.info(f"EPISODE_START: Episode {self.total_episodes + 1} 开始 - 全局步骤 {self.step_count + 1}")

        # 提取碰撞信息
        collision = self.extract_collision_info(info)
        
        # 更新episode指标
        self.episode_steps += 1
        self.step_count += 1
        
        # 检测碰撞
        if collision:
            self.episode_had_collision = True
            logger.info(f"COLLISION_DEBUG: 全局步骤 {self.step_count}, episode步骤 {self.episode_steps} - 碰撞发生！info={info}")
        
        # 计算基于episode长度的cost和reward修正（含CVaR）
        # Base Cost: 碰撞时给予高cost，鼓励避免碰撞
        if collision:
            base_cost = self.config['collision_penalty']  # 碰撞高惩罚
        else:
            base_cost = 0.0  # 无碰撞无cost
        
        # CVaR Cost: 基于风险管理的额外cost
        cvar_cost = self.calculate_cvar_cost(self.episode_steps)
        step_cost = base_cost + cvar_cost
        
        # Reward修正: 每步给予小额奖励，鼓励更长的episode
        step_reward_bonus = self.config['step_reward_weight']
        modified_reward = reward + step_reward_bonus
        
        # Episode结束时的处理
        if done:
            if not self.episode_had_collision:
                # 安全完成episode的奖励
                length_bonus = self.episode_steps * 0.1
                modified_reward += length_bonus
                logger.info(f"EPISODE_END: 安全完成episode，长度: {self.episode_steps}，长度奖励: {length_bonus:.2f}")
            else:
                logger.info(f"EPISODE_END: 碰撞结束episode，长度: {self.episode_steps}")
            
            # CVaR Penalty: 对worst case episode给予额外惩罚 (仅当CVaR启用时)
            cvar_penalty = self.calculate_cvar_penalty(self.episode_steps, done)
            if cvar_penalty > 0:
                modified_reward -= cvar_penalty
                logger.info(f"CVAR_PENALTY: Episode属于worst case，CVaR惩罚: {cvar_penalty:.2f}")
        
        # 调试输出（含CVaR信息）
        if self.step_count % self.config['log_interval'] == 0:
            import numpy as np
            collision_rate = self.total_collision_episodes / self.total_episodes if self.total_episodes > 0 else 0.0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            # 构建日志信息
            log_msg = (f"EPISODE_PROGRESS: 全局步骤 {self.step_count}, episode步骤 {self.episode_steps} - "
                      f"当前episode碰撞: {self.episode_had_collision}, "
                      f"总体碰撞率: {collision_rate:.4f}, 平均episode长度: {avg_length:.1f}")
            
            # 如果启用CVaR，添加CVaR信息
            if self.enable_cvar:
                cvar_stats = self.get_cvar_statistics()
                cvar_threshold = cvar_stats['cvar_threshold']
                log_msg += f", CVaR阈值: {cvar_threshold:.1f}"
            
            logger.info(log_msg)
            
            # 保存到TensorBoard (步级指标)
            if self.writer is not None:
                self.writer.add_scalar('training/collision_rate_running', collision_rate, self.step_count)
                self.writer.add_scalar('training/step_cost_total', step_cost, self.step_count)
                self.writer.add_scalar('training/step_cost_base', base_cost, self.step_count)
                self.writer.add_scalar('training/step_reward_modified', modified_reward, self.step_count)
                self.writer.add_scalar('training/episode_length_running_avg', avg_length, self.step_count)
                self.writer.add_scalar('training/current_episode_steps', self.episode_steps, self.step_count)
                
                # CVaR指标 (仅当启用时)
                if self.enable_cvar and 'cvar_stats' in locals():
                    self.writer.add_scalar('training/step_cost_cvar', cvar_cost, self.step_count)
                    self.writer.add_scalar('cvar/threshold', cvar_threshold, self.step_count)
                    self.writer.add_scalar('cvar/worst_case_rate', cvar_stats['worst_case_rate'], self.step_count)

        # Episode结束处理  
        if done:
            self.end_episode()
            
        # 调用父类方法，传递修正的reward和cost给BFTQ
        return super().record(state, action, modified_reward, next_state, done, 
                            {**info, 'cost': step_cost})

    def end_episode(self):
        """Episode结束时的处理"""
        if self.episode_steps == 0:
            return
        
        # 检查当前episode是否发生碰撞，更新碰撞episode计数
        if self.episode_had_collision:
            self.total_collision_episodes += 1
            logger.info(f"EPISODE_DEBUG: 碰撞episode计数增加 - total_collision_episodes: {self.total_collision_episodes}")
        else:
            logger.info(f"EPISODE_DEBUG: 安全episode - total_collision_episodes保持: {self.total_collision_episodes}")
        
        # 更新episode总数
        self.total_episodes += 1
        
        # 记录episode长度
        self.episode_lengths.append(self.episode_steps)
        
        # 计算新的碰撞率：碰撞episode数 / 总episode数
        collision_rate = self.total_collision_episodes / self.total_episodes
        
        # 调试输出
        logger.info(f"EPISODE_DEBUG: Episode {self.total_episodes} 结束 - "
                   f"episode步数: {self.episode_steps}, "
                   f"当前episode碰撞: {self.episode_had_collision}, "
                   f"碰撞episode总数: {self.total_collision_episodes}, "
                   f"总episode数: {self.total_episodes}, "
                   f"碰撞率: {collision_rate:.4f}")
        
        # 保存到TensorBoard（避免与Evaluation的episode/length重复，含CVaR指标）
        if self.writer is not None:
            self.writer.add_scalar('collision/episode_collision_rate', collision_rate, self.total_episodes)
            self.writer.add_scalar('collision/episode_had_collision', int(self.episode_had_collision), self.total_episodes)
            
            # 记录累积统计
            import numpy as np
            if len(self.episode_lengths) > 0:
                avg_length = np.mean(self.episode_lengths)
                max_length = np.max(self.episode_lengths)
                min_length = np.min(self.episode_lengths)
                self.writer.add_scalar('collision/avg_episode_length', avg_length, self.total_episodes)
                self.writer.add_scalar('collision/max_episode_length', max_length, self.total_episodes)
                self.writer.add_scalar('collision/min_episode_length', min_length, self.total_episodes)
                
                # CVaR指标 (仅当启用时)
                if self.enable_cvar:
                    cvar_stats = self.get_cvar_statistics()
                    self.writer.add_scalar('cvar/episode_threshold', cvar_stats['cvar_threshold'], self.total_episodes)
                    self.writer.add_scalar('cvar/episode_value', cvar_stats['cvar_value'], self.total_episodes)
                    self.writer.add_scalar('cvar/episode_worst_case_rate', cvar_stats['worst_case_rate'], self.total_episodes)
                    
                    # 标记当前episode是否为worst case
                    is_worst_case = self.episode_steps <= cvar_stats['cvar_threshold'] and cvar_stats['cvar_threshold'] > 0
                    self.writer.add_scalar('cvar/episode_is_worst_case', int(is_worst_case), self.total_episodes)
        
        # 保存到CSV
        if self.csv_writer is not None:
            self.csv_writer.writerow({
                'episode': self.total_episodes,
                'collision_rate': collision_rate,
                'episode_length': self.episode_steps,
                'had_collision': int(self.episode_had_collision)
            })
            self.csv_file.flush()
        
        # 重置episode指标
        self.episode_steps = 0
        self.episode_had_collision = False
        
        logger.info(f"EPISODE_DEBUG: Episode指标已重置 - episode_had_collision: {self.episode_had_collision}")
        
        # 每50个episode输出episode长度统计并记录训练进展
        if self.total_episodes % 50 == 0:
            self.print_episode_length_stats()
            # 记录训练进展指标到TensorBoard
            if self.writer is not None and len(self.episode_lengths) >= 20:
                import numpy as np
                recent_avg = np.mean(self.episode_lengths[-20:])  # 最近20个episode
                early_avg = np.mean(self.episode_lengths[:20])   # 最初20个episode
                improvement = recent_avg - early_avg
                self.writer.add_scalar('training/episode_length_improvement', improvement, self.total_episodes)
                
                # 记录长episode比例
                avg_length = np.mean(self.episode_lengths)
                long_episodes = sum(1 for length in self.episode_lengths if length > avg_length)
                long_episode_rate = long_episodes / len(self.episode_lengths) if len(self.episode_lengths) > 0 else 0
                self.writer.add_scalar('training/long_episode_rate', long_episode_rate, self.total_episodes)

    def get_collision_statistics(self) -> Dict[str, Any]:
        """获取当前碰撞统计信息"""
        if self.total_episodes == 0:
            return {'collision_rate': 0.0, 'total_episodes': 0, 'total_collision_episodes': 0}
            
        collision_rate = self.total_collision_episodes / self.total_episodes
        
        return {
            'collision_rate': collision_rate,
            'episode_length': self.episode_steps,
            'total_episodes': self.total_episodes,
            'total_collision_episodes': self.total_collision_episodes,
            'current_episode_had_collision': self.episode_had_collision
        }

    def print_training_summary(self):
        """打印训练摘要"""
        if self.total_episodes == 0:
            logger.info("尚未完成任何episode")
            return
            
        import numpy as np
        collision_rate = self.total_collision_episodes / self.total_episodes
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        max_length = np.max(self.episode_lengths) if self.episode_lengths else 0
        min_length = np.min(self.episode_lengths) if self.episode_lengths else 0
        
        logger.info(f"训练摘要:")
        logger.info(f"  总Episodes: {self.total_episodes}")
        logger.info(f"  碰撞Episodes: {self.total_collision_episodes}")
        logger.info(f"  碰撞率: {collision_rate:.4f} ({self.total_collision_episodes}/{self.total_episodes})")
        logger.info(f"  平均Episode长度: {avg_length:.2f}")
        logger.info(f"  最长Episode: {max_length} 步")
        logger.info(f"  最短Episode: {min_length} 步")
        
        # 显示episode长度趋势
        if len(self.episode_lengths) >= 20:
            recent_avg = np.mean(self.episode_lengths[-20:])  # 最近20个episode
            early_avg = np.mean(self.episode_lengths[:20])   # 最初20个episode
            improvement = recent_avg - early_avg
            logger.info(f"  Episode长度改进: {improvement:+.2f} (最近20个 vs 最初20个)")
        
        # 显示长episode的比例（超过平均长度的episode）
        if avg_length > 0:
            long_episodes = sum(1 for length in self.episode_lengths if length > avg_length)
            long_episode_rate = long_episodes / len(self.episode_lengths)
            logger.info(f"  长Episode比例: {long_episode_rate:.2f} (超过平均长度的episode)")
        
        # CVaR风险管理统计 (仅当启用时)
        if self.enable_cvar:
            cvar_stats = self.get_cvar_statistics()
            if cvar_stats['cvar_threshold'] > 0:
                logger.info(f"CVaR风险管理统计 (α={self.cvar_alpha}):")
                logger.info(f"  CVaR阈值: {cvar_stats['cvar_threshold']:.1f} 步")
                logger.info(f"  CVaR值: {cvar_stats['cvar_value']:.1f} 步 (worst case平均长度)")
                logger.info(f"  Worst case比例: {cvar_stats['worst_case_rate']:.3f} ({cvar_stats['worst_case_episodes']}/{len(self.episode_lengths)})")
                
                # 计算CVaR改进
                if len(self.episode_lengths) >= 40:
                    early_lengths = self.episode_lengths[:20]
                    recent_lengths = self.episode_lengths[-20:]
                    
                    early_worst = [l for l in early_lengths if l <= np.percentile(early_lengths, self.cvar_alpha * 100)]
                    recent_worst = [l for l in recent_lengths if l <= np.percentile(recent_lengths, self.cvar_alpha * 100)]
                    
                    if early_worst and recent_worst:
                        early_cvar = np.mean(early_worst)
                        recent_cvar = np.mean(recent_worst)
                        cvar_improvement = recent_cvar - early_cvar
                        logger.info(f"  CVaR改进: {cvar_improvement:+.2f} 步 (worst case性能提升)")
        else:
            logger.info("CVaR风险管理: 未启用")

    def print_episode_length_stats(self):
        """打印episode长度统计"""
        if not self.episode_lengths:
            return
            
        import numpy as np
        lengths = np.array(self.episode_lengths)
        
        logger.info(f"EPISODE_LENGTH_STATS: 总episodes: {len(lengths)}")
        logger.info(f"EPISODE_LENGTH_STATS: 平均长度: {np.mean(lengths):.2f}")
        logger.info(f"EPISODE_LENGTH_STATS: 最短长度: {np.min(lengths)}")
        logger.info(f"EPISODE_LENGTH_STATS: 最长长度: {np.max(lengths)}")
        logger.info(f"EPISODE_LENGTH_STATS: 标准差: {np.std(lengths):.2f}")
        
        # 长度分布
        unique, counts = np.unique(lengths, return_counts=True)
        length_dist = dict(zip(unique, counts))
        logger.info(f"EPISODE_LENGTH_STATS: 长度分布: {length_dist}")

    def close(self):
        """关闭资源"""
        if self.csv_file:
            self.csv_file.close()
        if hasattr(super(), 'close'):
            super().close()


# CollisionAwareBFTQ类结束 