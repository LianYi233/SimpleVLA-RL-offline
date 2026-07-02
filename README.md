# 环境安装
```
# 参照 SimpleVLA-RL-README.md说明，完成verl、LIBERO和OpenVLA-OFT相关安装与数据下载，这部分坑比较多，安装有问题的话随时@zhuoxu
```

# 运行脚本
```
bash src/examples/run_openvla_oft_rl_libero_offline_grpo.sh 
```

# 核心代码简要说明
```
# src 包含了所有针对offline rollout适配的代码，包括主运行loop、rollout、以及reward
```

## src/main_ppo
```
# 主程序运行入口，其中RobRewardManager为Reward函数调用入口
```

## src/ray_trainer
```
# 主运行loop所在，fit函数为核心training loop入口。其中包含rollout，log_prob计算，reward计算，advantage计算，以及policy update等关键步骤
```

## src/rob_rollout
```
# offline改动的核心所在，关键点在于rollout过程不再与环境进行交互，随机采样一个step，然后train过程只进行一次actions的生成，并使用RobRewardManager中设计的reward函数对这个action chunk进行打分，而非依赖最终的结果。

# 关键阅读 982 行 # training generate 以下部分
```

## src/core_alogs
```
# 关于grpo算法的核心函数
```
