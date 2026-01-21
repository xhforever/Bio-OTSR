"""
Bio-OTSR关键BUG修复补丁
应用此补丁可立即修复骨架对齐问题

使用方法:
1. 备份原文件
2. 运行此脚本自动应用补丁: python HOTFIX_PATCH.py
"""

import os

# ============= 修复 #1: Basis Matrix索引 =============
FIX_1 = {
    "file": "lib/body_models/skel/otsr_solver.py",
    "line": 240,
    "old": "            bm_indices = self.a_parent_idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)",
    "new": "            bm_indices = self.a_child_idx.view(1, -1, 1, 1).expand(B, -1, 3, 3)",
    "reason": "Ra对应当前关节的A-pose变换,不是父关节"
}

# ============= 修复 #2: TYPE_A关节索引 =============
FIX_2 = {
    "file": "lib/body_models/skel/kin_skel.py",
    "line": 216,
    "old": """    'TYPE_A': {
        'femur_r':   {'child': 2,  'parent': 1,  'params': [3, 4, 5]},   # Hip R
        'femur_l':   {'child': 7,  'parent': 6,  'params': [10, 11, 12]}, # Hip L
        'humerus_r': {'child': 16, 'parent': 15, 'params': [29, 30, 31]}, # Shoulder R
        'humerus_l': {'child': 21, 'parent': 20, 'params': [39, 40, 41]}  # Shoulder L
    },""",
    "new": """    'TYPE_A': {
        'femur_r':   {'child': 1,  'parent': 0,  'params': [3, 4, 5]},   # Hip R: pelvis → femur_r
        'femur_l':   {'child': 6,  'parent': 0,  'params': [10, 11, 12]}, # Hip L: pelvis → femur_l
        'humerus_r': {'child': 15, 'parent': 12, 'params': [29, 30, 31]}, # Shoulder R: thorax → humerus_r
        'humerus_l': {'child': 20, 'parent': 12, 'params': [39, 40, 41]}  # Shoulder L: thorax → humerus_l
    },""",
    "reason": "关节索引应指向正确的父子关系:髋关节=pelvis→femur,肩关节=thorax→humerus"
}

def apply_patch():
    """自动应用补丁"""
    import shutil
    from pathlib import Path
    
    # 确保在SKEL-CF目录下
    if not os.path.exists("lib/body_models/skel/otsr_solver.py"):
        print("❌ 错误: 请在SKEL-CF项目根目录下运行此脚本!")
        return
    
    print("🔧 开始应用Bio-OTSR修复补丁...")
    print("=" * 60)
    
    # 修复 #1
    print(f"\n📝 修复 #1: {FIX_1['file']}")
    print(f"   原因: {FIX_1['reason']}")
    file_path = FIX_1['file']
    backup_path = file_path + ".backup"
    
    # 备份
    shutil.copy(file_path, backup_path)
    print(f"   ✅ 已备份到: {backup_path}")
    
    # 读取并替换
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if FIX_1['old'] in content:
        content = content.replace(FIX_1['old'], FIX_1['new'])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ 已修复: a_parent_idx → a_child_idx")
    else:
        print(f"   ⚠️  未找到目标代码,可能已修复或版本不匹配")
    
    # 修复 #2
    print(f"\n📝 修复 #2: {FIX_2['file']}")
    print(f"   原因: {FIX_2['reason']}")
    file_path = FIX_2['file']
    backup_path = file_path + ".backup"
    
    # 备份
    if not os.path.exists(backup_path):
        shutil.copy(file_path, backup_path)
        print(f"   ✅ 已备份到: {backup_path}")
    
    # 读取并替换
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "'child': 2,  'parent': 1," in content:
        content = content.replace(
            "'child': 2,  'parent': 1,",
            "'child': 1,  'parent': 0,"
        )
        content = content.replace(
            "'child': 7,  'parent': 6,",
            "'child': 6,  'parent': 0,"
        )
        content = content.replace(
            "'child': 16, 'parent': 15,",
            "'child': 15, 'parent': 12,"
        )
        content = content.replace(
            "'child': 21, 'parent': 20,",
            "'child': 20, 'parent': 12,"
        )
        
        # 更新注释
        content = content.replace("# Hip R", "# Hip R: pelvis → femur_r")
        content = content.replace("# Hip L", "# Hip L: pelvis → femur_l")
        content = content.replace("# Shoulder R", "# Shoulder R: thorax → humerus_r")
        content = content.replace("# Shoulder L", "# Shoulder L: thorax → humerus_l")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ 已修复: TYPE_A关节索引映射")
    else:
        print(f"   ⚠️  未找到目标代码,可能已修复或版本不匹配")
    
    print("\n" + "=" * 60)
    print("✅ 补丁应用完成!")
    print("\n📋 修复摘要:")
    print("  1. Basis Matrix索引: a_parent_idx → a_child_idx")
    print("  2. TYPE_A关节索引: 修正为正确的父子关系")
    print("\n🔄 下一步:")
    print("  1. 重新训练或测试模型")
    print("  2. 对比修复前后的骨架对齐效果")
    print("  3. 如需回滚,使用备份文件(.backup)")

if __name__ == "__main__":
    apply_patch()

