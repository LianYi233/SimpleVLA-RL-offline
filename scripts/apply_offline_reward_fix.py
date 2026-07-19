#!/usr/bin/env python3
"""Apply offline GRPO reward fixes to SimpleVLA-RL-offline source tree."""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/2604-VLA_RL_offline/SimpleVLA_RL_Offline-main"
)


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"Expected block not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"updated {path}")


def main() -> None:
    if not (TARGET / "src").is_dir():
        raise SystemExit(f"Target not found: {TARGET}")

    replace(
        TARGET / "src/rob_rollout.py",
        '''def check_action_match(pred, gt, step):
    chunk_len = pred.shape[0] 

    gt_windows = gt.unfold(0, chunk_len, 1).transpose(1, 2)
    
    # gt_windows = trajectory_deltas(gt_windows)
    # pred = trajectory_deltas(pred)

    diffs = pred.unsqueeze(0) - gt_windows
    errors = torch.norm(diffs, dim=2).mean(dim=1)

    match_idx = errors.argmin()
    match_error = errors[step]

    is_match = torch.abs(match_idx - step) <= 1
    return is_match, match_idx, match_error
    

def batch_check_action_match(pred, gt, step):
    """
    Args:
        pred: [B, 8, 7]
        gt: List of [T, 7]
    Returns:
        match_flags: [B] bool tensor
        match_positions: [B] int tensor (-1 表示未匹配)
        match_scores: [B] float tensor (最佳匹配的误差)
    """
    batch_size = pred.shape[0]
    device = pred.device
    chunk_len = pred.shape[1]
    
    match_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    match_positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    match_scores = torch.full((batch_size,), float('inf'), device=device)
    step = torch.tensor(step, device=device)
    
    for i in range(batch_size):
        pred_seq = pred[i]  # [8, 7]
        gt_seq = gt[i]  
        step_i = step[i]    # [T_i, 7]
        
        is_match, match_idx, match_error = check_action_match(pred_seq, gt_seq, step_i)
        
        match_flags[i] = is_match
        match_positions[i] = match_idx
        match_scores[i] = match_error

    # reward = torch.exp(-match_scores) * match_flags + torch.clamp(1 - torch.abs(match_positions - step) / (chunk_len - 2), min=0)
    reward = torch.exp(-match_scores) * match_flags + match_flags.float()
    
    return match_flags, reward''',
        '''def check_action_match(pred, gt, step, action_std=None, match_tolerance=2):
    """Score a predicted action chunk against expert trajectory windows.

    Returns a continuous reward in roughly [0, 1.5] so GRPO always receives
    non-zero variance within a prompt group, instead of hard zeroing on mismatch.
    """
    chunk_len = pred.shape[0]
    num_windows = gt.shape[0] - chunk_len + 1
    if num_windows <= 0:
        zero = torch.tensor(0.0, device=pred.device)
        return False, torch.tensor(0, device=pred.device), torch.tensor(float('inf'), device=pred.device), zero

    gt_windows = gt.unfold(0, chunk_len, 1).transpose(1, 2)

    diffs = pred.unsqueeze(0) - gt_windows
    if action_std is not None:
        std = action_std.view(1, 1, -1).clamp(min=1e-6)
        diffs = diffs / std

    errors = torch.norm(diffs, dim=2).mean(dim=1)

    step_idx = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
    step_idx = max(0, min(step_idx, errors.shape[0] - 1))

    step_error = errors[step_idx]
    match_idx = errors.argmin()
    position_offset = torch.abs(match_idx - step_idx).float()

    action_reward = torch.exp(-step_error.clamp(max=10.0))
    max_offset = max(chunk_len - 1, 1)
    alignment_reward = torch.clamp(1.0 - position_offset / max_offset, min=0.0)
    best_window_reward = torch.exp(-errors[match_idx].clamp(max=10.0))

    reward = action_reward + 0.3 * alignment_reward + 0.2 * best_window_reward
    is_match = position_offset <= match_tolerance
    return is_match, match_idx, step_error, reward


def batch_check_action_match(pred, gt, step, action_std=None, match_tolerance=2):
    """
    Args:
        pred: [B, chunk_len, action_dim]
        gt: List of [T, action_dim]
        step: List[int] or tensor of sampled trajectory indices
        action_std: optional [action_dim] normalization stats for scale-invariant error
    Returns:
        match_flags: [B] bool tensor (soft temporal alignment within tolerance)
        reward: [B] float tensor, continuous in roughly [0, 1.5]
    """
    batch_size = pred.shape[0]
    device = pred.device

    match_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
    step = torch.tensor(step, device=device)

    for i in range(batch_size):
        is_match, _, _, reward = check_action_match(
            pred[i],
            gt[i],
            step[i],
            action_std=action_std,
            match_tolerance=match_tolerance,
        )
        match_flags[i] = is_match
        rewards[i] = reward

    return match_flags, rewards''',
    )

    replace(
        TARGET / "src/rob_rollout.py",
        '''            for data in batch_data:
                end_idx = len(data['full_image']) - n_samples - self.config.action_chunks_len - 1
                step_idx = random.randint(0, end_idx) # random step''',
        '''            action_norm_stats = self.module.norm_stats[self.config.unnorm_key]["action"]
            action_std = torch.tensor(
                action_norm_stats["std"], device=device, dtype=torch.float32
            )

            for data in batch_data:
                traj_len = len(data['actions'])
                end_idx = max(0, traj_len - self.config.action_chunks_len - 1)
                step_idx = random.randint(0, end_idx) if end_idx > 0 else 0''',
    )

    replace(
        TARGET / "src/rob_rollout.py",
        '''            match_flags, match_rewards = batch_check_action_match(
                torch.from_numpy(actions).to(device), gt_actions, input_steps)''',
        '''            match_flags, match_rewards = batch_check_action_match(
                torch.from_numpy(actions).to(device),
                gt_actions,
                input_steps,
                action_std=action_std,
            )''',
    )

    replace(
        TARGET / "src/core_algos.py",
        '''def get_score_mean_std(id2score):
    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:               
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    return id2mean, id2std''',
        '''def get_score_mean_std(id2score, min_std=0.1):
    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            scores = torch.tensor(id2score[idx], dtype=torch.float32)
            id2mean[idx] = scores.mean()
            id2std[idx] = scores.std().clamp(min=min_std)
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    return id2mean, id2std''',
    )

    replace(
        TARGET / "src/core_algos.py",
        '''def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):''',
        '''def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   grpo_min_std: float = 0.1):''',
    )

    replace(
        TARGET / "src/core_algos.py",
        '''        for i in range(bsz):
            id2score[index[i]].append(scores[i])                         
        id2mean, id2std = get_score_mean_std(id2score)

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # scores[i] = scores[i] - id2mean[index[i]]''',
        '''        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        id2mean, id2std = get_score_mean_std(id2score, min_std=grpo_min_std)

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)''',
    )

    replace(
        TARGET / "src/main_ppo.py",
        '''        score = [float(item) for item in completes]''',
        '''        if 'match_rewards' in data.batch:
            score = data.batch['match_rewards'].cpu().numpy().tolist()
        else:
            score = [float(item) for item in completes]''',
    )

    replace(
        TARGET / "src/ray_trainer.py",
        '''        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)''',
        '''        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            grpo_min_std=config.algorithm.adv_params.get('grpo_min_std', 0.1),
        )''',
    )

    yaml_path = TARGET / "src/config/ppo_trainer.yaml"
    yaml = yaml_path.read_text(encoding="utf-8")
    needle = "    reward_model_gamma: ${algorithm.gamma}\n"
    insert = needle + "    grpo_min_std: 0.1  # floor for GRPO std normalization to avoid gradient blow-up\n"
    if "grpo_min_std" not in yaml:
        if needle not in yaml:
            raise RuntimeError(f"Expected YAML block not found in {yaml_path}")
        yaml_path.write_text(yaml.replace(needle, insert, 1), encoding="utf-8")
        print(f"updated {yaml_path}")
    else:
        print(f"skip {yaml_path} (already patched)")

    print(f"\nAll fixes applied under: {TARGET}")


if __name__ == "__main__":
    main()
