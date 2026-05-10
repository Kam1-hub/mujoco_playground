# G1 人形机器人训练环境

从 MuJoCo Playground 提取的宇树 G1 人形机器人完整训练 pipeline。

## 特性

- 完整的 G1 环境定义（平坦地形、粗糙地形）
- MJX 加速仿真
- 域随机化支持
- JAX PPO 训练（Brax）
- RSL-RL 训练（PyTorch）

## 安装

```bash
cd template
pip install -e .
```

如果需要 CUDA 支持：
```bash
pip install -e ".[cuda]"
```

如果需要 RSL-RL 训练支持：
```bash
pip install -e ".[learning]"
```

## 快速开始

### JAX PPO 训练

```bash
# 平坦地形训练
train-g1-jax --env_name G1JoystickFlatTerrain --num_timesteps 200000000

# 粗糙地形训练
train-g1-jax --env_name G1JoystickRoughTerrain --num_timesteps 200000000

# 使用域随机化
train-g1-jax --env_name G1JoystickFlatTerrain --domain_randomization
```

### RSL-RL 训练

```bash
# 平坦地形训练
train-g1-rsl --env_name G1JoystickFlatTerrain --num_envs 4096

# 粗糙地形训练
train-g1-rsl --env_name G1JoystickRoughTerrain --num_envs 4096
```

## 环境说明

- **G1JoystickFlatTerrain**: 平坦地形上的手柄控制任务
  - 观测维度: 101 (state) / 150+ (privileged_state)
  - 动作维度: 29 (关节位置命令)
  - 控制频率: 50Hz
  - Episode 长度: 1000 步 (20 秒)

- **G1JoystickRoughTerrain**: 粗糙地形上的手柄控制任务
  - 与平坦地形相同的配置
  - 增加了地形高度场和纹理

## 项目结构

```
template/
├── g1_env/                     # 核心环境包
│   ├── __init__.py
│   ├── _src/                   # 环境实现
│   │   ├── mjx_env.py          # MJX 环境基类
│   │   ├── gait.py             # 步态生成工具
│   │   ├── reward.py           # 奖励函数工具
│   │   ├── registry.py         # 环境注册表
│   │   ├── wrapper.py          # Brax 包装器
│   │   ├── wrapper_torch.py    # Torch 包装器
│   │   └── locomotion/
│   │       └── g1/             # G1 环境定义
│   │           ├── base.py
│   │           ├── g1_constants.py
│   │           ├── joystick.py
│   │           ├── randomize.py
│   │           └── xmls/       # MuJoCo 模型文件
│   └── config/                 # RL 训练配置
│       └── locomotion_params.py
├── learning/                   # 训练脚本
│   ├── train_jax_ppo.py        # JAX PPO 训练
│   └── train_rsl_rl.py         # RSL-RL 训练
├── pyproject.toml              # 项目配置
├── .gitignore
└── README.md
```

## 训练配置

### Brax PPO 默认配置

- 总时间步: 200M
- 评估次数: 20
- 并行环境数: 8192
- 网络架构: Policy (512, 256, 128), Value (512, 256, 128)
- 学习率: 3e-4
- 熵成本: 0.005
- 裁剪参数: 0.2

### RSL-RL 默认配置

- 最大迭代次数: 1000
- 每环境步数: 24
- 网络架构: Actor/Critic (512, 256, 128)
- 学习率: 3e-4
- 裁剪参数: 0.2

## 依赖

- Python >= 3.11
- JAX
- MuJoCo >= 3.6.0
- MuJoCo MJX >= 3.6.0
- Brax >= 0.14.2
- RSL-RL >= 3.0.0（可选，用于 RSL-RL 训练）

## 许可证

Apache 2.0（继承自 MuJoCo Playground）

## 致谢

本项目提取自 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)，感谢 Google DeepMind 团队的开源贡献。
