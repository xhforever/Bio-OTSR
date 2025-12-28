import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# ================= 配置区域 =================
# 1. 这里填写你截图中那个包含3个文件的文件夹路径
SOURCE_LOG_DIR = "/data/yangxianghao/SKEL-CF/data_outputs/exp/1226-TRAIN-p40-4gpu-finetune-h36m-coco/tensorboards" 

# 2. 这里填写合并后输出的文件夹路径（运行后在TensorBoard指向这个新目录）
OUTPUT_LOG_DIR = "/data/yangxianghao/SKEL-CF/data_outputs/exp/1226-TRAIN-p40-4gpu-finetune-h36m-coco/tensorboards/merged_tensorboard_log"

# 3. 模式选择：
# "append": 适用于Step被重置的情况（如文件1是0-4000，文件2也是0-4000，合并后变成0-8000）
# "overwrite": 适用于Step未重置但有重叠的情况（如文件1是0-4000，文件2是3000-7000，合并后以文件2为准）
MERGE_MODE = "append" 
# ===========================================

def extract_events(log_dir):
    # 获取目录下所有 tfevents 文件，并按时间戳排序
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    # 按照文件名中的时间戳排序，确保顺序正确
    files.sort(key=lambda x: int(x.split('.')[-4]) if x.split('.')[-4].isdigit() else x)
    
    print(f"检测到 {len(files)} 个日志文件，顺序如下：")
    for f in files:
        print(f" -> {os.path.basename(f)}")
        
    return files

def fix_tensorboard(source_dir, output_dir, mode="append"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir=output_dir)
    files = extract_events(source_dir)
    
    global_step_offset = 0
    last_file_max_step = 0
    
    # 用于记录上一个文件的最后一个step，防止完全重复
    
    for i, file_path in enumerate(files):
        print(f"正在处理文件: {os.path.basename(file_path)} ...")
        
        # 加载事件数据
        ea = EventAccumulator(file_path)
        ea.Reload()
        
        # 获取所有标量标签 (Loss, LR, etc.)
        tags = ea.Tags()['scalars']
        
        # 临时存储当前文件的最大step，用于更新offset
        current_file_max_step = 0
        
        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                original_step = event.step
                value = event.value
                
                # === 核心逻辑：计算新的 Step ===
                if mode == "append":
                    # 累加模式：当前step + 之前所有文件的最大step总和
                    new_step = original_step + global_step_offset
                else:
                    # 覆盖模式：直接使用原始step（自动去重由TensorBoard处理，或者这里加逻辑）
                    new_step = original_step
                
                writer.add_scalar(tag, value, new_step)
                
                # 记录当前文件的最大step
                if original_step > current_file_max_step:
                    current_file_max_step = original_step

        # 处理完一个文件后，更新 Offset
        if mode == "append":
            # 如果是append模式，下一个文件的0应当从当前文件的max_step + 1开始
            # 或者是简单地加上当前文件的跨度
            print(f"  -> 文件结束，原始 Max Step: {current_file_max_step}")
            print(f"  -> 当前累计 Offset: {global_step_offset}")
            
            # 更新 Offset：加上当前文件的最大步数（假设下一个文件从0开始）
            # 注意：如果下一个文件不是从0开始，而是一定程度的重叠，这里需要更复杂的逻辑。
            # 针对你描述的“前两个都训练了4398step”，假设它们都是从0-4398，则：
            global_step_offset += (current_file_max_step + 1) # +1 避免步数重复

    print("========================================")
    print(f"合并完成！请运行: tensorboard --logdir {output_dir}")
    writer.close()

if __name__ == "__main__":
    # 为了防止路径错误，建议使用绝对路径，或者确保相对路径正确
    # 你需要把图片里的那三个文件放到 source_dir 指向的文件夹里
    fix_tensorboard(SOURCE_LOG_DIR, OUTPUT_LOG_DIR, mode=MERGE_MODE)