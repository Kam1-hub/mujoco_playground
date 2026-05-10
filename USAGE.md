# G1 训练环境使用指南

## 快速开始

### 1. 安装依赖

```bash
cd D:/mujoco_playground/template

# 基础安装
pip install -e .

# 如果需要 CUDA 支持
pip install -e ".[cuda]"

# 如果需要 RSL-RL 训练
pip install -e ".[learning]"

# 完整安装（包含所有可选依赖）
pip install -e ".[all]"
```

### 2. 验证安装

```bash
# 测试导入
python -c "import g1_env; print('可用环境:', g1_env.registry.ALL_ENVS)"

# 应该输出: 可用环境: ('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')
```

### 3. 开始训练

#### JAX PPO 训练（推荐）

```bash
# 平坦地形，快速测试（100万步）
train-g1-jax --env_name G1JoystickFlatTerrain --num_timesteps 1000000

# 平坦地形，完整训练（2亿步）
train-g1-jax --env_name G1JoystickFlatTerrain --num_timesteps 200000000

# 粗糙地形 + 域随机化
train-g1-jax --env_name G1JoystickRoughTerrain --domain_randomization --num_timesteps 200000000

# 使用 WandB 记录
train-g1-jax --env_name G1JoystickFlatTerrain --use_wandb

# 自定义日志目录
train-g1-jax --env_name G1JoystickFlatTerrain --logdir ./my_logs
```

#### RSL-RL 训练（PyTorch）

```bash
# 平坦地形训练
train-g1-rsl --env_name G1JoystickFlatTerrain --num_envs 4096

# 粗糙地形训练
train-g1-rsl --env_name G1JoystickRoughTerrain --num_envs 4096

# 多 GPU 训练
train-g1-rsl --env_name G1JoystickFlatTerrain --multi_gpu

# 从检查点恢复
train-g1-rsl --env_name G1JoystickFlatTerrain --load_run_name my_run --checkpoint_num 100
```

### 4. 训练参数说明

#### 常用参数

- `--env_name`: 环境名称（G1JoystickFlatTerrain 或 G1JoystickRoughTerrain）
- `--num_timesteps`: 总训练步数（JAX PPO）
- `--num_envs`: 并行环境数量
- `--domain_randomization`: 启用域随机化
- `--use_wandb`: 使用 WandB 记录
- `--logdir`: 日志保存目录
- `--seed`: 随机种子

#### 高级参数

- `--learning_rate`: 学习率（默认 3e-4）
- `--entropy_cost`: 熵成本（默认 0.005）
- `--num_evals`: 评估次数（默认 20）
- `--policy_hidden_layer_sizes`: 策略网络层大小
- `--value_hidden_layer_sizes`: 价值网络层大小

## 环境说明

### G1JoystickFlatTerrain

- **任务**: 在平坦地形上跟随手柄命令移动
- **观测维度**: 101 (state) / 150+ (privileged_state)
- **动作维度**: 29 (关节位置命令)
- **控制频率**: 50Hz
- **Episode 长度**: 1000 步 (20 秒)

### G1JoystickRoughTerrain

- **任务**: 在粗糙地形上跟随手柄命令移动
- **配置**: 与平坦地形相同
- **额外挑战**: 地形高度场和纹理变化

## 训练配置

### Brax PPO 默认配置

```python
num_timesteps = 200_000_000
num_evals = 20
num_envs = 8192
learning_rate = 3e-4
entropy_cost = 0.005
clipping_epsilon = 0.2
network = {
    "policy": (512, 256, 128),
    "value": (512, 256, 128)
}
```

### RSL-RL 默认配置

```python
max_iterations = 1000
num_steps_per_env = 24
learning_rate = 3e-4
clip_param = 0.2
network = {
    "actor": [512, 256, 128],
    "critic": [512, 256, 128]
}
```

## 常见问题

### 1. 导入错误

```
ModuleNotFoundError: No module named 'jax'
```

**解决**: 确保已安装依赖 `pip install -e .`

### 2. CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决**: 减少并行环境数量 `--num_envs 2048`

### 3. 训练速度慢

**优化建议**:
- 使用 CUDA 版本的 JAX: `pip install -e ".[cuda]"`
- 增加并行环境数: `--num_envs 16384`
- 使用 Warp 后端: `--impl warp`

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
├── README.md                   # 项目说明
└── USAGE.md                    # 本文件
```

## 进一步学习

1. **修改奖励函数**: 编辑 `g1_env/_src/locomotion/g1/joystick.py` 中的 `_get_reward()` 方法
2. **调整观测空间**: 修改 `_get_obs()` 方法
3. **添加新任务**: 参考 `joystick.py` 创建新的任务类
4. **自定义域随机化**: 编辑 `g1_env/_src/locomotion/g1/randomize.py`

## 许可证

Apache 2.0（继承自 MuJoCo Playground）
