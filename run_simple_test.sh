#!/bin/bash
#
# Bio-OTSR修复简化验证脚本
# 不依赖完整数据集，只验证修复是否生效
#

echo "🔧 Bio-OTSR修复验证"
echo "===================="
echo ""

# 激活conda环境
if [ -d "$HOME/miniconda3/envs/skelvit" ]; then
    echo "📦 激活conda环境: skelvit"
    source $HOME/miniconda3/bin/activate skelvit
elif [ -d "$HOME/anaconda3/envs/skelvit" ]; then
    echo "📦 激活conda环境: skelvit"
    source $HOME/anaconda3/bin/activate skelvit
else
    echo "⚠️  未找到skelvit环境，使用当前Python环境"
fi

echo ""

# 运行测试
if [ -f "test_fix_simple.py" ]; then
    python test_fix_simple.py "$@"
else
    echo "❌ 错误: 未找到test_fix_simple.py"
    echo "   请在SKEL-CF目录下运行此脚本"
    exit 1
fi

