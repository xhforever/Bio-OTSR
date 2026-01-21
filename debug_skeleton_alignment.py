"""
骨架对齐调试脚本
用于可视化Bio-OTSR预测的骨架与图像特征的对齐情况

使用方法:
    python debug_skeleton_alignment.py --img demo_images/test.jpg --checkpoint path/to/model.pth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

# SKEL关节名称和层次结构
SKEL_JOINTS = [
    'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
    'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l',
    'lumbar_body', 'thorax', 'head',
    'scapula_r', 'humerus_r', 'ulna_r', 'radius_r', 'hand_r',
    'scapula_l', 'humerus_l', 'ulna_l', 'radius_l', 'hand_l'
]

# 关节连接(用于绘制骨架)
SKEL_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # 右腿
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), # 左腿
    (0, 11), (11, 12), (12, 13),             # 脊柱+头
    (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), # 右臂
    (12, 19), (19, 20), (20, 21), (21, 22), (22, 23)  # 左臂
]

# TYPE_A关节(修复前后的配置)
TYPE_A_OLD = {
    'femur_r': (2, 1),   # ❌ 错误: tibia_r ← femur_r
    'femur_l': (7, 6),   # ❌ 错误: tibia_l ← femur_l
    'humerus_r': (16, 15), # ❌ 错误: ulna_r ← humerus_r
    'humerus_l': (21, 20), # ❌ 错误: ulna_l ← humerus_l
}

TYPE_A_NEW = {
    'femur_r': (1, 0),   # ✅ 正确: femur_r ← pelvis
    'femur_l': (6, 0),   # ✅ 正确: femur_l ← pelvis
    'humerus_r': (15, 12), # ✅ 正确: humerus_r ← thorax
    'humerus_l': (20, 12), # ✅ 正确: humerus_l ← thorax
}

def visualize_skeleton_3d(pred_kp3d, title="Predicted Skeleton", type_a_config=None):
    """
    可视化3D骨架
    
    Args:
        pred_kp3d: (24, 3) numpy array
        title: 图表标题
        type_a_config: TYPE_A关节配置,用于高亮显示
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有关节点
    ax.scatter(pred_kp3d[:, 0], pred_kp3d[:, 1], pred_kp3d[:, 2], 
               c='blue', s=50, alpha=0.6, label='Joints')
    
    # 绘制骨架连接
    for i, j in SKEL_SKELETON:
        ax.plot([pred_kp3d[i, 0], pred_kp3d[j, 0]],
                [pred_kp3d[i, 1], pred_kp3d[j, 1]],
                [pred_kp3d[i, 2], pred_kp3d[j, 2]],
                'gray', linewidth=1, alpha=0.5)
    
    # 高亮TYPE_A关节(如果提供)
    if type_a_config:
        for name, (child, parent) in type_a_config.items():
            # 绘制高亮的骨骼向量
            ax.plot([pred_kp3d[parent, 0], pred_kp3d[child, 0]],
                    [pred_kp3d[parent, 1], pred_kp3d[child, 1]],
                    [pred_kp3d[parent, 2], pred_kp3d[child, 2]],
                    'red', linewidth=3, alpha=0.8, label=name)
            
            # 标注关节名称
            mid_x = (pred_kp3d[parent, 0] + pred_kp3d[child, 0]) / 2
            mid_y = (pred_kp3d[parent, 1] + pred_kp3d[child, 1]) / 2
            mid_z = (pred_kp3d[parent, 2] + pred_kp3d[child, 2]) / 2
            ax.text(mid_x, mid_y, mid_z, name, fontsize=8)
    
    # 标注特殊关节
    important_joints = [0, 11, 12]  # pelvis, lumbar, thorax
    for idx in important_joints:
        ax.text(pred_kp3d[idx, 0], pred_kp3d[idx, 1], pred_kp3d[idx, 2],
                SKEL_JOINTS[idx], fontsize=10, color='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # 设置相同的坐标轴比例
    max_range = np.array([pred_kp3d[:, 0].max()-pred_kp3d[:, 0].min(),
                          pred_kp3d[:, 1].max()-pred_kp3d[:, 1].min(),
                          pred_kp3d[:, 2].max()-pred_kp3d[:, 2].min()]).max() / 2.0
    
    mid_x = (pred_kp3d[:, 0].max()+pred_kp3d[:, 0].min()) * 0.5
    mid_y = (pred_kp3d[:, 1].max()+pred_kp3d[:, 1].min()) * 0.5
    mid_z = (pred_kp3d[:, 2].max()+pred_kp3d[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig

def compare_type_a_vectors(pred_kp3d):
    """
    对比修复前后TYPE_A关节的骨骼向量
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TYPE_A关节向量对比 (修复前 vs 修复后)', fontsize=16, fontweight='bold')
    
    for idx, (name, (old_child, old_parent)) in enumerate(TYPE_A_OLD.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        new_child, new_parent = TYPE_A_NEW[name]
        
        # 计算向量
        old_vec = pred_kp3d[old_child] - pred_kp3d[old_parent]
        new_vec = pred_kp3d[new_child] - pred_kp3d[new_parent]
        
        # 计算向量长度
        old_len = np.linalg.norm(old_vec)
        new_len = np.linalg.norm(new_vec)
        
        # 绘制向量
        ax.quiver(0, 0, 0, old_vec[0], old_vec[1], old_vec[2], 
                  color='red', label=f'修复前: {SKEL_JOINTS[old_parent]}→{SKEL_JOINTS[old_child]}',
                  arrow_length_ratio=0.1, linewidth=2)
        
        ax.quiver(0, 0, 0, new_vec[0], new_vec[1], new_vec[2],
                  color='green', label=f'修复后: {SKEL_JOINTS[new_parent]}→{SKEL_JOINTS[new_child]}',
                  arrow_length_ratio=0.1, linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name}\n修复前长度:{old_len:.3f} | 修复后长度:{new_len:.3f}')
        ax.legend()
        
        # 设置3D视图
        ax = plt.subplot(2, 2, idx+1, projection='3d')
        ax.quiver(0, 0, 0, old_vec[0], old_vec[1], old_vec[2], 
                  color='red', label=f'❌ {SKEL_JOINTS[old_parent]}→{SKEL_JOINTS[old_child]}',
                  arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(0, 0, 0, new_vec[0], new_vec[1], new_vec[2],
                  color='green', label=f'✅ {SKEL_JOINTS[new_parent]}→{SKEL_JOINTS[new_child]}',
                  arrow_length_ratio=0.1, linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{name}\n修复前长度:{old_len:.3f} | 修复后长度:{new_len:.3f}')
        ax.legend()
    
    plt.tight_layout()
    return fig

def check_fk_consistency(pred_kp3d):
    """
    检查3D关节坐标是否满足FK约束
    """
    print("\n" + "="*60)
    print("FK一致性检查")
    print("="*60)
    
    # 检查骨骼长度合理性
    print("\n📏 骨骼长度统计:")
    bone_lengths = {}
    for i, j in SKEL_SKELETON:
        length = np.linalg.norm(pred_kp3d[j] - pred_kp3d[i])
        bone_name = f"{SKEL_JOINTS[i]} → {SKEL_JOINTS[j]}"
        bone_lengths[bone_name] = length
        print(f"  {bone_name:30s}: {length:.4f}")
    
    # 检查左右对称性
    print("\n⚖️  左右对称性检查:")
    symmetry_pairs = [
        (1, 6, "femur"),   # 大腿
        (2, 7, "tibia"),   # 小腿
        (15, 20, "humerus"), # 上臂
        (16, 21, "ulna"),    # 前臂
    ]
    
    for r_idx, l_idx, name in symmetry_pairs:
        r_vec = pred_kp3d[r_idx] - pred_kp3d[0]  # 相对pelvis
        l_vec = pred_kp3d[l_idx] - pred_kp3d[0]
        
        r_len = np.linalg.norm(r_vec)
        l_len = np.linalg.norm(l_vec)
        diff = abs(r_len - l_len)
        ratio = diff / max(r_len, l_len) * 100
        
        status = "✅" if ratio < 5 else "⚠️" if ratio < 10 else "❌"
        print(f"  {status} {name:10s}: 右={r_len:.4f} | 左={l_len:.4f} | 差异={ratio:.2f}%")
    
    # 检查TYPE_A关节向量的合理性
    print("\n🎯 TYPE_A关节检查 (修复后的配置):")
    for name, (child, parent) in TYPE_A_NEW.items():
        vec = pred_kp3d[child] - pred_kp3d[parent]
        length = np.linalg.norm(vec)
        print(f"  {name:12s}: {SKEL_JOINTS[parent]:12s} → {SKEL_JOINTS[child]:12s} | 长度={length:.4f}")

def main():
    parser = argparse.ArgumentParser(description='骨架对齐调试工具')
    parser.add_argument('--dummy', action='store_true', 
                        help='使用虚拟数据进行演示')
    args = parser.parse_args()
    
    # 生成虚拟的3D关节数据(或从模型加载)
    if args.dummy:
        print("🔧 使用虚拟数据进行演示...")
        # 构造一个标准T-pose骨架
        pred_kp3d = np.zeros((24, 3))
        pred_kp3d[0] = [0, 0, 0]      # pelvis
        pred_kp3d[1] = [0.1, -0.1, 0] # femur_r
        pred_kp3d[2] = [0.1, -0.5, 0] # tibia_r
        pred_kp3d[6] = [-0.1, -0.1, 0] # femur_l
        pred_kp3d[7] = [-0.1, -0.5, 0] # tibia_l
        pred_kp3d[11] = [0, 0.2, 0]    # lumbar
        pred_kp3d[12] = [0, 0.4, 0]    # thorax
        pred_kp3d[13] = [0, 0.6, 0]    # head
        pred_kp3d[15] = [0.2, 0.4, 0]  # humerus_r
        pred_kp3d[16] = [0.4, 0.4, 0]  # ulna_r
        pred_kp3d[20] = [-0.2, 0.4, 0] # humerus_l
        pred_kp3d[21] = [-0.4, 0.4, 0] # ulna_l
    else:
        # TODO: 从实际模型推理结果加载
        print("❌ 尚未实现从模型加载,请使用 --dummy 参数")
        return
    
    # 可视化修复前的配置
    print("\n📊 生成可视化图表...")
    fig1 = visualize_skeleton_3d(pred_kp3d, 
                                  "修复前的TYPE_A配置 (❌ 错误)", 
                                  TYPE_A_OLD)
    fig1.savefig('debug_skeleton_old.png', dpi=150, bbox_inches='tight')
    print("  ✅ 已保存: debug_skeleton_old.png")
    
    # 可视化修复后的配置
    fig2 = visualize_skeleton_3d(pred_kp3d,
                                  "修复后的TYPE_A配置 (✅ 正确)",
                                  TYPE_A_NEW)
    fig2.savefig('debug_skeleton_new.png', dpi=150, bbox_inches='tight')
    print("  ✅ 已保存: debug_skeleton_new.png")
    
    # 对比TYPE_A向量
    fig3 = compare_type_a_vectors(pred_kp3d)
    fig3.savefig('debug_type_a_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✅ 已保存: debug_type_a_comparison.png")
    
    # FK一致性检查
    check_fk_consistency(pred_kp3d)
    
    print("\n✅ 调试完成! 请查看生成的PNG文件")
    plt.show()

if __name__ == "__main__":
    main()

