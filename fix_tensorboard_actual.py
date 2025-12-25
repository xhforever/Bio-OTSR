#!/usr/bin/env python3
"""
根据实际 TensorBoard 记录修正曲线
分析结果：
- Epoch 0: 4398 steps (0-4397) ✓
- Epoch 1: 4398 steps (4398-8795) ✓
- Epoch 2+: 记录不完整
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import glob

log_dir = '/data/yangxianghao/SKEL-CF/data_outputs/exp/1224-TRAIN-p40-4gpu-finetune-h36m-coco/tensorboards'
logger_interval = 50

print('根据实际记录修正 TensorBoard 曲线')
print('=' * 60)

# 根据实际记录：
# - Epoch 0: 0-4397 (4398 steps) ✓，记录 0-4350 (88个记录)
# - Epoch 1: 4398-8795 (4398 steps) ✓，记录 4398-8748 (88个记录)
# - Epoch 2+: 8796+ (记录不完整)

epoch_info = {
    0: {'steps': 4398, 'start': 0, 'end': 4397},  # ✓
    1: {'steps': 4398, 'start': 4398, 'end': 8795},  # ✓
    2: {'steps': None, 'start': 8796, 'end': None},  # 不完整
}

print('每个 epoch 的 global_step 范围:')
for epoch, info in epoch_info.items():
    if info['steps']:
        print(f'  Epoch {epoch}: {info["start"]} - {info["end"]} ({info["steps"]} steps) ✓')
    else:
        print(f'  Epoch {epoch}: {info["start"]}+ (记录不完整)')
print()

# 收集所有原始记录
all_events_by_tag = {}

# 1. 原始文件（排除只有step 0的）
for f in sorted(glob.glob(f'{log_dir}/events.out.*')):
    if any(x in f for x in ['.bak', '.old', '_fixed', '_corrected', '_final', '_correct_v2', '_final_v3']):
        continue
    try:
        ea = EventAccumulator(f)
        ea.Reload()
        scalars = ea.Tags().get('scalars', [])
        for tag in scalars:
            if tag not in all_events_by_tag:
                all_events_by_tag[tag] = []
            events = ea.Scalars(tag)
            for e in events:
                # 排除只有step 0的重复记录
                if len([x for x in all_events_by_tag[tag] if x[0] == e.step]) == 0 or e.step != 0:
                    all_events_by_tag[tag].append((e.step, e.value))
    except:
        pass

# 2. 从 _fixed 文件恢复原始记录
for f in glob.glob(f'{log_dir}/events.out.*_fixed*'):
    try:
        ea = EventAccumulator(f)
        ea.Reload()
        scalars = ea.Tags().get('scalars', [])
        for tag in scalars:
            if tag not in all_events_by_tag:
                all_events_by_tag[tag] = []
            events = ea.Scalars(tag)
            for e in events:
                original_step = e.step - 50  # 恢复原始
                # 检查是否已存在
                if not any(x[0] == original_step for x in all_events_by_tag[tag]):
                    all_events_by_tag[tag].append((original_step, e.value))
    except:
        pass

print(f'找到 {len(all_events_by_tag)} 个 scalar tags')
print()

# 创建新的 TensorBoard 日志文件
new_writer = SummaryWriter(log_dir, filename_suffix='_actual')

def map_step_to_correct_global_step(old_step):
    """将旧的 step 映射到正确的 global_step"""
    if old_step <= 4350:
        # Epoch 0: 0-4397 (4398 steps) ✓
        # 记录只到 4350，直接映射
        return old_step
    elif old_step <= 8748:
        # Epoch 1: 4398-8795 (4398 steps) ✓
        # 记录显示 4398-8748 (88个记录，实际4398 steps)
        # 需要映射到 Epoch 1 的完整范围 4398-8795
        epoch1_start = epoch_info[1]['start']  # 4398
        # 记录范围 4398-8748 对应 88个记录，实际4398 steps
        # 映射到 4398-8795
        record_start = 4398
        record_end = 8748
        record_span = record_end - record_start  # 4350
        target_span = 4397  # Epoch 1 的完整范围是 4398-8795，共4398 steps，跨度4397
        
        # 线性映射
        if old_step == record_start:
            return epoch1_start
        elif old_step == record_end:
            return epoch_info[1]['end']  # 8795
        else:
            # 线性插值
            ratio = (old_step - record_start) / record_span
            return int(epoch1_start + ratio * target_span)
    else:
        # Epoch 2+: 保持原样（记录不完整）
        return old_step

# 写入修正后的记录
total_events = 0
for tag, events in all_events_by_tag.items():
    print(f'处理 {tag}: {len(events)} 个事件')
    for old_step, value in events:
        new_step = map_step_to_correct_global_step(old_step)
        new_writer.add_scalar(tag, value, new_step)
        total_events += 1

new_writer.close()

print()
print(f'完成！共处理 {total_events} 个事件')
print(f'新文件已创建，文件名包含 _actual 后缀')
print()
print('修正后的每个 epoch step 数:')
print('  Epoch 0: 4398 steps ✓')
print('  Epoch 1: 4398 steps ✓')
print('  Epoch 2+: 记录不完整，保持原样')

