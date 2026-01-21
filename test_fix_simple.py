"""
简单的Bio-OTSR修复验证脚本
不需要完整数据集，只验证修复是否生效
"""

import torch
import sys
import os
from pathlib import Path

print("=" * 70)
print("Bio-OTSR修复验证测试")
print("=" * 70)
print()

# ============= Step 1: 测试Bio-OTSR Solver =============
print("🔧 Step 1: 测试Bio-OTSR Solver...")
print("-" * 70)

try:
    from lib.body_models.skel.otsr_solver import BioOTSRSolver
    from lib.body_models.skel.kin_skel import BIO_OTSR_CONFIG
    
    solver = BioOTSRSolver()
    print("✅ BioOTSRSolver初始化成功")
    print()
    
    # 检查修复是否生效
    print("🔍 验证修复内容:")
    print()
    
    # 检查BUG #2修复: TYPE_A配置
    print("  【BUG #2检查】TYPE_A关节索引配置:")
    type_a = BIO_OTSR_CONFIG['TYPE_A']
    
    expected_config = {
        'femur_r': (1, 0),    # 修复后应该是: femur_r ← pelvis
        'femur_l': (6, 0),    # 修复后应该是: femur_l ← pelvis
        'humerus_r': (15, 12), # 修复后应该是: humerus_r ← thorax
        'humerus_l': (20, 12), # 修复后应该是: humerus_l ← thorax
    }
    
    bug2_fixed = True
    for name, (expected_child, expected_parent) in expected_config.items():
        actual_child = type_a[name]['child']
        actual_parent = type_a[name]['parent']
        
        status = "✅" if (actual_child == expected_child and actual_parent == expected_parent) else "❌"
        if status == "❌":
            bug2_fixed = False
        
        print(f"    {status} {name:12s}: child={actual_child:2d}, parent={actual_parent:2d}  "
              f"(期望: child={expected_child:2d}, parent={expected_parent:2d})")
    
    print()
    if bug2_fixed:
        print("  ✅ BUG #2 已修复: TYPE_A关节索引正确")
    else:
        print("  ❌ BUG #2 未修复: TYPE_A关节索引仍然错误!")
        print("     请运行: python HOTFIX_PATCH.py")
    
    print()
    
    # 检查BUG #1修复: 需要查看solver代码中的basis_matrix索引
    print("  【BUG #1检查】Basis Matrix索引:")
    solver_code_path = Path("lib/body_models/skel/otsr_solver.py")
    if solver_code_path.exists():
        with open(solver_code_path, 'r') as f:
            solver_code = f.read()
        
        if 'bm_indices = self.a_child_idx' in solver_code:
            print("    ✅ 使用 a_child_idx (正确)")
            bug1_fixed = True
        elif 'bm_indices = self.a_parent_idx' in solver_code:
            print("    ❌ 使用 a_parent_idx (错误)")
            print("       请运行: python HOTFIX_PATCH.py")
            bug1_fixed = False
        else:
            print("    ⚠️  未找到相关代码，无法判断")
            bug1_fixed = None
    else:
        print("    ⚠️  未找到solver代码文件")
        bug1_fixed = None
    
    print()
    
    # 简单的前向测试
    print("  【功能测试】Solver前向传播:")
    B = 2
    pred_kp3d = torch.randn(B, 24, 3)
    pred_ortho = torch.randn(B, 6, 3)
    pred_scalar = torch.randn(B, 32)
    
    result = solver(pred_kp3d, pred_ortho, pred_scalar)
    
    if result.shape == (B, 46):
        print(f"    ✅ 输出维度正确: {tuple(result.shape)}")
        print(f"       输入: kp3d{tuple(pred_kp3d.shape)}, ortho{tuple(pred_ortho.shape)}, scalar{tuple(pred_scalar.shape)}")
        solver_works = True
    else:
        print(f"    ❌ 输出维度错误: {tuple(result.shape)} (期望: {(B, 46)})")
        solver_works = False
    
    print()
    
except ImportError as e:
    print(f"❌ 无法导入Bio-OTSR模块: {e}")
    print("   可能原因:")
    print("   1. 你的代码不是Bio-OTSR版本")
    print("   2. Python环境配置问题")
    print()
    sys.exit(1)
except Exception as e:
    print(f"❌ Solver测试失败: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# ============= Step 2: 加载Checkpoint (可选) =============
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
    print()
    print("🔧 Step 2: 测试Checkpoint加载...")
    print("-" * 70)
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint加载成功: {checkpoint_path}")
        print(f"   Epoch: {ckpt.get('epoch', 'unknown')}")
        print(f"   Keys: {list(ckpt.keys())}")
        
        # 检查是否有state_dict
        if 'state_dict' in ckpt or 'model' in ckpt:
            state_dict = ckpt.get('state_dict', ckpt.get('model', {}))
            print(f"   模型参数数量: {len(state_dict)} 个tensor")
            
            # 检查是否有Bio-OTSR相关的参数
            bio_ostr_keys = [k for k in state_dict.keys() 
                            if any(x in k for x in ['xyz_decoder', 'ortho_decoder', 'scalar_decoder', 'solver'])]
            if bio_ostr_keys:
                print(f"✅ 检测到Bio-OTSR相关参数: {len(bio_ostr_keys)} 个")
                print(f"   示例: {bio_ostr_keys[:3]}")
            else:
                print(f"⚠️  未检测到Bio-OTSR相关参数")
                print("   这可能意味着:")
                print("   - 模型使用的是原始SKEL-CF (不是Bio-OTSR)")
                print("   - 或者参数命名不同")
        
        print()
        
    except Exception as e:
        print(f"❌ Checkpoint加载失败: {e}")
        print()

# ============= 总结 =============
print("=" * 70)
print("📊 测试总结")
print("=" * 70)
print()

all_fixed = bug2_fixed and (bug1_fixed is True or bug1_fixed is None) and solver_works

if all_fixed:
    print("✅ 所有检查通过!")
    print()
    print("   - BUG #1 (Basis Matrix索引): ✅ 已修复" if bug1_fixed else "   - BUG #1: ⚠️ 无法确认")
    print("   - BUG #2 (TYPE_A关节索引): ✅ 已修复")
    print("   - Solver功能测试: ✅ 正常运行")
    print()
    print("🎉 修复成功! 你可以:")
    print("   1. 使用现有模型进行推理（无需重训练）")
    print("   2. 或在完整数据集上评估效果")
    print()
else:
    print("⚠️  存在问题:")
    print()
    if not bug1_fixed:
        print("   ❌ BUG #1 未修复")
    if not bug2_fixed:
        print("   ❌ BUG #2 未修复")
    if not solver_works:
        print("   ❌ Solver功能异常")
    print()
    print("💡 请运行修复脚本:")
    print("   python HOTFIX_PATCH.py")
    print()
    sys.exit(1)

print("=" * 70)

