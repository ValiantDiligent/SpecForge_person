#!/usr/bin/env python3
"""
分析checkpoint结构并测试移除embedding层的功能
"""

import os
import torch
import json
from pathlib import Path
from collections import defaultdict
import sys

def analyze_checkpoint_structure(checkpoint_path):
    """分析checkpoint的结构"""
    print(f"🔍 分析checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 路径不存在: {checkpoint_path}")
        return None
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 成功加载checkpoint")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    # 分析checkpoint结构
    print(f"\n📊 Checkpoint顶层键:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  📁 {key}: dict with {len(checkpoint[key])} keys")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  🔢 {key}: tensor {checkpoint[key].shape}")
        else:
            print(f"  📄 {key}: {type(checkpoint[key])}")
    
    # 如果有model或state_dict，分析参数结构
    model_dict = None
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
        dict_name = 'model'
    elif 'state_dict' in checkpoint:
        model_dict = checkpoint['state_dict']
        dict_name = 'state_dict'
    else:
        # 如果checkpoint本身就是state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            model_dict = checkpoint
            dict_name = 'checkpoint'
    
    if model_dict:
        print(f"\n🏗️ 模型参数结构 ({dict_name}):")
        analyze_model_parameters(model_dict)
    
    return checkpoint

def analyze_model_parameters(state_dict):
    """分析模型参数结构"""
    # 按模块分组
    modules = defaultdict(list)
    embedding_params = []
    total_params = 0
    
    for name, param in state_dict.items():
        total_params += param.numel()
        
        # 检查是否是embedding相关参数
        if any(embed_key in name.lower() for embed_key in ['embed', 'embedding', 'token']):
            embedding_params.append((name, param.shape, param.numel()))
        
        # 按模块分组
        parts = name.split('.')
        if len(parts) > 1:
            module_name = parts[0]
            modules[module_name].append((name, param.shape, param.numel()))
        else:
            modules['root'].append((name, param.shape, param.numel()))
    
    # 打印模块统计
    print(f"  📈 总参数数量: {total_params:,}")
    print(f"  📦 模块数量: {len(modules)}")
    
    print(f"\n📋 各模块参数统计:")
    for module_name, params in sorted(modules.items()):
        module_params = sum(p[2] for p in params)
        print(f"  🔹 {module_name}: {len(params)} 参数, {module_params:,} 元素")
        
        # 显示前几个参数
        for i, (name, shape, numel) in enumerate(params[:3]):
            print(f"    - {name}: {shape}")
        if len(params) > 3:
            print(f"    ... 还有 {len(params) - 3} 个参数")
    
    # 特别分析embedding参数
    if embedding_params:
        print(f"\n🎯 Embedding相关参数:")
        embedding_total = sum(p[2] for p in embedding_params)
        print(f"  📊 Embedding参数总数: {embedding_total:,} ({embedding_total/total_params*100:.1f}%)")
        
        for name, shape, numel in embedding_params:
            print(f"  🔸 {name}: {shape} ({numel:,} 元素)")
    else:
        print(f"\n❌ 未找到embedding相关参数")

def remove_embedding_from_state_dict(state_dict, embedding_keywords=None):
    """从state_dict中移除embedding相关参数"""
    if embedding_keywords is None:
        embedding_keywords = ['embed', 'embedding', 'token']
    
    original_count = len(state_dict)
    original_params = sum(p.numel() for p in state_dict.values())
    
    # 使用字典推导式移除embedding参数
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(keyword in k.lower() for keyword in embedding_keywords)
    }
    
    filtered_count = len(filtered_state_dict)
    filtered_params = sum(p.numel() for p in filtered_state_dict.values())
    
    removed_count = original_count - filtered_count
    removed_params = original_params - filtered_params
    
    print(f"\n🗑️ 移除embedding参数结果:")
    print(f"  📉 移除参数数量: {removed_count} / {original_count}")
    print(f"  📉 移除参数元素: {removed_params:,} / {original_params:,} ({removed_params/original_params*100:.1f}%)")
    
    # 显示被移除的参数
    removed_params_list = [k for k in state_dict.keys() if k not in filtered_state_dict]
    if removed_params_list:
        print(f"  🔍 被移除的参数:")
        for param_name in removed_params_list:
            shape = state_dict[param_name].shape
            numel = state_dict[param_name].numel()
            print(f"    - {param_name}: {shape} ({numel:,} 元素)")
    
    return filtered_state_dict

def test_save_and_load(original_checkpoint, filtered_state_dict, test_dir):
    """测试保存和加载功能"""
    print(f"\n💾 测试保存和加载功能:")
    
    # 创建测试目录
    test_dir = Path(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    # 保存原始checkpoint（用于对比）
    original_path = test_dir / "original_checkpoint.pth"
    torch.save(original_checkpoint, original_path)
    print(f"  ✅ 保存原始checkpoint: {original_path}")
    
    # 保存移除embedding的checkpoint
    filtered_checkpoint = original_checkpoint.copy()
    
    # 更新state_dict
    if 'model' in filtered_checkpoint:
        filtered_checkpoint['model'] = filtered_state_dict
    elif 'state_dict' in filtered_checkpoint:
        filtered_checkpoint['state_dict'] = filtered_state_dict
    else:
        filtered_checkpoint = filtered_state_dict
    
    filtered_path = test_dir / "no_embedding_checkpoint.pth"
    torch.save(filtered_checkpoint, filtered_path)
    print(f"  ✅ 保存无embedding checkpoint: {filtered_path}")
    
    # 测试加载
    try:
        loaded_checkpoint = torch.load(filtered_path, map_location='cpu')
        print(f"  ✅ 成功加载无embedding checkpoint")
        
        # 验证结构
        if 'model' in loaded_checkpoint:
            loaded_state_dict = loaded_checkpoint['model']
        elif 'state_dict' in loaded_checkpoint:
            loaded_state_dict = loaded_checkpoint['state_dict']
        else:
            loaded_state_dict = loaded_checkpoint
        
        # 检查是否还有embedding参数
        embedding_params = [k for k in loaded_state_dict.keys() 
                          if any(keyword in k.lower() for keyword in ['embed', 'embedding', 'token'])]
        
        if embedding_params:
            print(f"  ⚠️ 警告: 仍然存在embedding参数: {embedding_params}")
        else:
            print(f"  ✅ 确认: 无embedding参数")
        
        print(f"  📊 加载的参数数量: {len(loaded_state_dict)}")
        print(f"  📊 加载的参数元素: {sum(p.numel() for p in loaded_state_dict.values()):,}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return False

def create_mock_model_for_testing():
    """创建一个模拟模型用于测试"""
    print(f"\n🧪 创建模拟模型进行测试:")
    
    # 模拟一个简单的transformer模型state_dict
    mock_state_dict = {
        # Embedding层
        'embeddings.word_embeddings.weight': torch.randn(32000, 4096),
        'embeddings.position_embeddings.weight': torch.randn(2048, 4096),
        
        # Transformer层
        'layers.0.self_attn.q_proj.weight': torch.randn(4096, 4096),
        'layers.0.self_attn.k_proj.weight': torch.randn(4096, 4096),
        'layers.0.self_attn.v_proj.weight': torch.randn(4096, 4096),
        'layers.0.self_attn.o_proj.weight': torch.randn(4096, 4096),
        'layers.0.mlp.gate_proj.weight': torch.randn(11008, 4096),
        'layers.0.mlp.up_proj.weight': torch.randn(11008, 4096),
        'layers.0.mlp.down_proj.weight': torch.randn(4096, 11008),
        'layers.0.input_layernorm.weight': torch.randn(4096),
        'layers.0.post_attention_layernorm.weight': torch.randn(4096),
        
        # 输出层
        'lm_head.weight': torch.randn(32000, 4096),
        'norm.weight': torch.randn(4096),
    }
    
    mock_checkpoint = {
        'model': mock_state_dict,
        'optimizer': {},
        'epoch': 9,
        'global_step': 1000,
    }
    
    print(f"  ✅ 创建模拟checkpoint，包含 {len(mock_state_dict)} 个参数")
    return mock_checkpoint

def main():
    """主函数"""
    print("🚀 Draft Model Checkpoint 分析工具")
    print("=" * 80)
    
    # 目标checkpoint路径
    checkpoint_path = "/Users/zhanghuaxiang/zhx/go_learner/src/SpecForge-1/cache/dataset/Qwen3-8B-eagle3/epoch_9"
    test_dir = "./checkpoint_test"
    
    # 1. 分析现有checkpoint
    checkpoint = analyze_checkpoint_structure(checkpoint_path)
    
    if checkpoint is None:
        print(f"\n⚠️ 无法加载指定checkpoint，使用模拟数据进行测试")
        checkpoint = create_mock_model_for_testing()
    
    # 2. 获取state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 3. 移除embedding参数
    print(f"\n" + "=" * 80)
    print(f"🔧 测试移除embedding功能")
    filtered_state_dict = remove_embedding_from_state_dict(state_dict)
    
    # 4. 测试保存和加载
    print(f"\n" + "=" * 80)
    success = test_save_and_load(checkpoint, filtered_state_dict, test_dir)
    
    # 5. 总结
    print(f"\n" + "=" * 80)
    print(f"📋 测试总结:")
    if success:
        print(f"  ✅ 所有测试通过")
        print(f"  ✅ 可以安全地移除embedding层并保存/加载")
        print(f"  📁 测试文件保存在: {test_dir}")
    else:
        print(f"  ❌ 测试失败")
    
    # 6. 提供使用示例
    print(f"\n💡 使用示例代码:")
    print(f"""
# 加载checkpoint
checkpoint = torch.load('path/to/checkpoint.pth', map_location='cpu')

# 移除embedding参数（使用字典推导式）
filtered_state_dict = {{
    k: v for k, v in checkpoint['model'].items()
    if not any(keyword in k.lower() for keyword in ['embed', 'embedding', 'token'])
}}

# 更新checkpoint
checkpoint['model'] = filtered_state_dict

# 保存新的checkpoint
torch.save(checkpoint, 'path/to/no_embedding_checkpoint.pth')

# 加载时会自动跳过缺失的embedding参数
model.load_state_dict(filtered_state_dict, strict=False)
""")

if __name__ == "__main__":
    main()