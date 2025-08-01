"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test) [options]
  experiments benchmark <benchmark> (--train|--test) [options]
  experiments -h | --help

Options:
  -h --help              Show this screen.
  --episodes <count>     Number of episodes [default: 5].
  --no-display           Disable environment, agent, and rewards rendering.
  --name-from-config     Name the output folder from the corresponding config files
  --processes <count>    Number of running processes [default: 4].
  --recover              Load model from the latest checkpoint.
  --recover-from <file>  Load model from a given checkpoint.
  --seed <str>           Seed the environments and agents.
  --train                Train the agent.
  --test                 Test the agent.
  --verbose              Set log level to debug instead of info.
  --repeat <times>       Repeat several times [default: 1].
  --with-metrics         Use collision-aware BFTQ with metrics tracking.
  --with-cvar            Enable CVaR risk management (requires --with-metrics).
  --cvar-alpha <float>   CVaR confidence level [default: 0.1].
"""
import datetime
import os
from pathlib import Path
import gymnasium as gym
import json
from docopt import docopt
from itertools import product
from multiprocessing.pool import Pool

from rl_agents.trainer import logger
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

# 导入碰撞感知功能
import sys
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from collision_aware_bftq import CollisionAwareBFTQ
    COLLISION_AWARE_AVAILABLE = True
except ImportError as e:
    COLLISION_AWARE_AVAILABLE = False
    print(f"Warning: Collision-aware BFTQ not available: {e}")

# Suppress matplotlib font manager debug messages early
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

BENCHMARK_FILE = 'benchmark_summary'
LOGGING_CONFIG = 'scripts/configs/logging.json'
VERBOSE_CONFIG = 'scripts/configs/verbose.json'


def load_collision_aware_agent(agent_config_path, env, enable_cvar=False, options=None):
    """
    加载碰撞感知BFTQ智能体
    
    :param agent_config_path: 智能体配置文件路径
    :param env: 环境对象
    :param enable_cvar: 是否启用CVaR风险管理
    :param options: 命令行选项字典
    :return: CollisionAwareBFTQ智能体
    """
    with open(agent_config_path, 'r') as f:
        agent_config = json.load(f)
    
    # 修改配置以使用CollisionAwareBFTQ
    original_class = agent_config.get("__class__", "")
    agent_config["__class__"] = "<class 'scripts.collision_aware_bftq.CollisionAwareBFTQ'>"
    
    # 设置CVaR相关配置
    agent_config["enable_cvar"] = enable_cvar
    
    # 处理CVaR alpha参数
    if options and '--cvar-alpha' in options and options['--cvar-alpha']:
        try:
            cvar_alpha = float(options['--cvar-alpha'])
            if 0.0 < cvar_alpha < 1.0:
                agent_config["cvar_alpha"] = cvar_alpha
                print(f"使用自定义CVaR α值: {cvar_alpha}")
            else:
                print(f"Warning: CVaR α值必须在(0,1)范围内，使用默认值0.1")
        except ValueError:
            print(f"Warning: 无效的CVaR α值 '{options['--cvar-alpha']}'，使用默认值0.1")
    
    print(f"原始智能体类: {original_class}")
    if enable_cvar:
        print(f"CVaR配置: α={agent_config.get('cvar_alpha', 0.1)}, "
              f"惩罚权重={agent_config.get('cvar_penalty_weight', 2.0)}, "
              f"最小episodes={agent_config.get('cvar_min_episodes', 10)}")
    
    # 创建智能体
    agent = CollisionAwareBFTQ(env, agent_config)
    
    return agent


def print_collision_statistics(agent):
    """
    打印碰撞统计信息
    
    :param agent: CollisionAwareBFTQ智能体
    """
    if not isinstance(agent, CollisionAwareBFTQ):
        return
    
    stats = agent.get_collision_statistics()
    print("\n" + "="*60)
    print("碰撞感知BFTQ训练统计信息")
    print("="*60)
    print(f"总轮数: {stats['total_episodes']}")
    print(f"碰撞轮数: {stats['total_collision_episodes']}")
    print(f"碰撞率: {stats['collision_rate']:.3f}")
    print(f"当前episode长度: {stats['episode_length']}")
    
    # 显示CVaR统计 (如果启用)
    if hasattr(agent, 'enable_cvar') and hasattr(agent, 'get_cvar_statistics'):
        if agent.enable_cvar:
            cvar_stats = agent.get_cvar_statistics()
            if cvar_stats['cvar_threshold'] > 0:
                print(f"\nCVaR风险管理 (α={agent.cvar_alpha*100:.0f}%):")
                print(f"  CVaR阈值: {cvar_stats['cvar_threshold']:.1f} 步")
                print(f"  CVaR值: {cvar_stats['cvar_value']:.1f} 步")
                print(f"  Worst case比例: {cvar_stats['worst_case_rate']:.3f}")
            else:
                print(f"\nCVaR风险管理: 已启用，等待足够数据 (需要{agent.cvar_min_episodes}个episodes)")
        else:
            print(f"\nCVaR风险管理: 未启用")
    
    print("="*60 + "\n")


def main():
    opts = docopt(__doc__)
    if opts['evaluate']:
        for _ in range(int(opts['--repeat'])):
            evaluate(opts['<environment>'], opts['<agent>'], opts)
    elif opts['benchmark']:
        benchmark(opts)


def evaluate(environment_config, agent_config, options):
    """
        Evaluate an agent interacting with an environment.

    :param environment_config: the path of the environment configuration file
    :param agent_config: the path of the agent configuration file
    :param options: the evaluation options
    """
    # Configure logger with default settings if config files don't exist
    if os.path.exists(LOGGING_CONFIG):
        logger.configure(LOGGING_CONFIG)
    else:
        logger.configure()  # Use default configuration
    
    if options['--verbose'] and os.path.exists(VERBOSE_CONFIG):
        logger.configure(VERBOSE_CONFIG)
    
    env = load_environment(environment_config)
    
    # 检查环境配置
    if hasattr(env, 'config'):
        print(f"环境配置: {env.config}")
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'config'):
        print(f"Environment unwrapped config: {env.unwrapped.config}")
    else:
        print("无法访问环境配置")
    
    # 检查是否使用碰撞感知模式和CVaR
    use_collision_aware = options.get('--with-metrics', False)
    use_cvar = options.get('--with-cvar', False)
    
    # CVaR需要collision-aware模式
    if use_cvar and not use_collision_aware:
        print("Warning: --with-cvar 需要同时指定 --with-metrics，自动启用 --with-metrics")
        use_collision_aware = True
    
    if use_collision_aware and COLLISION_AWARE_AVAILABLE:
        if use_cvar:
            print("使用碰撞感知BFTQ模式 + CVaR风险管理")
        else:
            print("使用碰撞感知BFTQ模式 (不含CVaR)")
        agent = load_collision_aware_agent(agent_config, env, enable_cvar=use_cvar, options=options)
    else:
        if use_collision_aware and not COLLISION_AWARE_AVAILABLE:
            print("Warning: --with-metrics参数被指定但碰撞感知模块不可用，使用标准代理")
        agent = load_agent(agent_config, env)
    
    run_directory = None
    if options['--name-from-config']:
        run_directory = "{}_{}_{}".format(Path(agent_config).with_suffix('').name,
                                  datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                                  os.getpid())
    options['--seed'] = int(options['--seed']) if options['--seed'] is not None else None
    evaluation = Evaluation(env,
                            agent,
                            run_directory=run_directory,
                            num_episodes=int(options['--episodes']),
                            sim_seed=options['--seed'],
                            recover=options['--recover'] or options['--recover-from'],
                            display_env=not options['--no-display'],
                            display_agent=not options['--no-display'],
                            display_rewards=not options['--no-display'])
    
    # 碰撞感知模式会通过agent.record方法自动处理，无需额外增强evaluation
    
    if options['--train']:
        evaluation.train()
        # 训练完成后打印碰撞统计信息
        if isinstance(agent, CollisionAwareBFTQ):
            print_collision_statistics(agent)
    elif options['--test']:
        evaluation.test()
        # 测试完成后打印碰撞统计信息
        if isinstance(agent, CollisionAwareBFTQ):
            print_collision_statistics(agent)
    else:
        evaluation.close()
    return os.path.relpath(evaluation.run_directory)


def benchmark(options):
    """
        Run the evaluations of several agents interacting in several environments.

    The evaluations are dispatched over several processes.
    The benchmark configuration file should look like this:
    {
        "environments": ["path/to/env1.json", ...],
        "agents: ["path/to/agent1.json", ...]
    }

    :param options: the evaluation options, containing the path to the benchmark configuration file.
    """
    # Prepare experiments
    with open(options['<benchmark>']) as f:
        benchmark_config = json.loads(f.read())
    generate_agent_configs(benchmark_config)
    experiments = product(benchmark_config['environments'], benchmark_config['agents'], [options])

    # Run evaluations
    with Pool(processes=int(options['--processes'])) as pool:
        results = pool.starmap(evaluate, experiments)

    # Clean temporary config files
    generate_agent_configs(benchmark_config, clean=True)

    # Write evaluations summary
    benchmark_filename = os.path.join(Evaluation.OUTPUT_FOLDER, '{}_{}.{}.json'.format(
        BENCHMARK_FILE, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid()))
    with open(benchmark_filename, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)
        gym.logger.info('Benchmark done. Summary written in: {}'.format(benchmark_filename))


def generate_agent_configs(benchmark_config, clean=False):
    """
        Generate several agent configurations from:
        - a "base_agent" configuration path field
        - a "key" field referring to a parameter that should vary
        - a "values" field listing the values of the parameter taken for each agent

        Created agent configurations will be stored in temporary file, that can be removed after use by setting the
        argument clean=True.
    :param benchmark_config: a benchmark configuration
    :param clean: should the temporary agent configurations files be removed
    :return the updated benchmark config
    """
    if "base_agent" in benchmark_config:
        with open(benchmark_config["base_agent"], 'r') as f:
            base_config = json.load(f)
            configs = [dict(base_config, **{benchmark_config["key"]: value})
                       for value in benchmark_config["values"]]
            paths = [Path(benchmark_config["base_agent"]).parent / "bench_{}={}.json".format(benchmark_config["key"], value)
                     for value in benchmark_config["values"]]
            if clean:
                [path.unlink() for path in paths]
            else:
                [json.dump(config, path.open('w')) for config, path in zip(configs, paths)]
            benchmark_config["agents"] = paths
    return benchmark_config


if __name__ == "__main__":
    main()
