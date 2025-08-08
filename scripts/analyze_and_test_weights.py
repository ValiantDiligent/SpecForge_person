#!/usr/bin/env python3
"""
分析和测试draft model权重的脚本
1. 查看权重结构
2. 移除embedding权重并保存
3. 测试加载功能
"""

import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_weights_structure(checkpoint_path):
    """分析权重文件结构"""
    print(f"\n=== 分析权重文件: {checkpoint_path} ===")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return None
    
    # 检查文件格式
    if checkpoint_path.endswith('.safetensors'):
        print("📁 文件格式: SafeTensors")
        return analyze_safetensors(checkpoint_path)
    elif checkpoint_path.endswith('.bin') or checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
        print("📁 文件格式: PyTorch")
        return analyze_pytorch_weights(checkpoint_path)
    else:
        print("❓ 未知文件格式")
        return None

def analyze_safetensors(file_path):
    """分析safetensors文件"""
    try:
        weights = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            print(f"🔍 总共有 {len(f.keys())} 个权重张量")
            print("\n📊 权重结构分析:")
            
            embedding_keys = []
            non_embedding_keys = []
            
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = tensor
                
                # 分类权重
                if any(embed_keyword in key.lower() for embed_keyword in ['embed', 'embedding', 'token']):
                    embedding_keys.append(key)
                else:
                    non_embedding_keys.append(key)
                
                print(f"  {key}: {tensor.shape} ({tensor.dtype})")
            
            print(f"\n🎯 Embedding相关权重 ({len(embedding_keys)} 个):")
            for key in embedding_keys:
                tensor = weights[key]
                print(f"  ✓ {key}: {tensor.shape} ({tensor.dtype})")
            
            print(f"\n🔧 非Embedding权重 ({len(non_embedding_keys)} 个):")
            for key in non_embedding_keys[:10]:  # 只显示前10个
                tensor = weights[key]
                print(f"  • {key}: {tensor.shape} ({tensor.dtype})")
            if len(non_embedding_keys) > 10:
                print(f"  ... 还有 {len(non_embedding_keys) - 10} 个权重")
        
        return weights, embedding_keys, non_embedding_keys
        
    except Exception as e:
        print(f"❌ 读取safetensors文件失败: {e}")
        return None

def analyze_pytorch_weights(file_path):
    """分析PyTorch权重文件"""
    try:
        weights = torch.load(file_path, map_location='cpu')
        print(f"🔍 总共有 {len(weights)} 个权重张量")
        print("\n📊 权重结构分析:")
        
        embedding_keys = []
        non_embedding_keys = []
        
        for key, tensor in weights.items():
            # 分类权重
            if any(embed_keyword in key.lower() for embed_keyword in ['embed', 'embedding', 'token']):
                embedding_keys.append(key)
            else:
                non_embedding_keys.append(key)
            
            print(f"  {key}: {tensor.shape} ({tensor.dtype})")
        
        print(f"\n🎯 Embedding相关权重 ({len(embedding_keys)} 个):")
        for key in embedding_keys:
            tensor = weights[key]
            print(f"  ✓ {key}: {tensor.shape} ({tensor.dtype})")
        
        print(f"\n🔧 非Embedding权重 ({len(non_embedding_keys)} 个):")
        for key in non_embedding_keys[:10]:  # 只显示前10个
            tensor = weights[key]
            print(f"  • {key}: {tensor.shape} ({tensor.dtype})")
        if len(non_embedding_keys) > 10:
            print(f"  ... 还有 {len(non_embedding_keys) - 10} 个权重")
        
        return weights, embedding_keys, non_embedding_keys
        
    except Exception as e:
        print(f"❌ 读取PyTorch文件失败: {e}")
        return None

def remove_embedding_and_save(weights, embedding_keys, non_embedding_keys, original_path):
    """移除embedding权重并保存"""
    print(f"\n=== 移除Embedding权重并保存 ===")
    
    # 创建不包含embedding的权重字典
    weights_no_embed = {}
    for key in non_embedding_keys:
        if isinstance(weights, dict):
            weights_no_embed[key] = weights[key]
        else:
            # 如果是safetensors，需要重新读取
            with safe_open(original_path, framework="pt", device="cpu") as f:
                weights_no_embed[key] = f.get_tensor(key)
    
    print(f"✂️ 移除了 {len(embedding_keys)} 个embedding权重")
    print(f"💾 保留了 {len(weights_no_embed)} 个非embedding权重")
    
    # 保存新的权重文件
    output_path = original_path.replace('.safetensors', '_no_embed.safetensors')
    if not output_path.endswith('_no_embed.safetensors'):
        output_path = original_path.replace('.bin', '_no_embed.safetensors')
        output_path = output_path.replace('.pt', '_no_embed.safetensors')
        output_path = output_path.replace('.pth', '_no_embed.safetensors')
    
    try:
        save_file(weights_no_embed, output_path)
        print(f"✅ 成功保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None

def test_loading_with_from_pretrained(no_embed_path):
    """测试使用from_pretrained加载无embedding权重"""
    print(f"\n=== 测试加载无Embedding权重 ===")
    
    try:
        # 导入必要的模块
        from specforge.modeling.auto import AutoEagle3DraftModel
        from specforge.modeling.auto import AutoDraftModelConfig
        
        # 加载配置
        config_path = "configs/qwen3-8b-eagle3.json"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        print(f"📋 加载配置文件: {config_path}")
        draft_model_config = AutoDraftModelConfig.from_file(config_path)
        
        # 尝试加载模型（模拟train_eagle3_online.py中的代码）
        print(f"🔄 尝试加载模型: {no_embed_path}")
        
        # 这里模拟第155-159行的代码
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(no_embed_path)
            .cuda() if torch.cuda.is_available() else AutoEagle3DraftModel.from_pretrained(no_embed_path)
        )
        
        if torch.cuda.is_available():
            draft_model = draft_model.to(torch.bfloat16)
        
        print("✅ 模型加载成功！")
        print(f"📊 模型参数数量: {sum(p.numel() for p in draft_model.parameters()):,}")
        
        # 检查embedding层状态
        print("\n🔍 检查embedding层状态:")
        for name, module in draft_model.named_modules():
            if 'embed' in name.lower():
                print(f"  {name}: {type(module)}")
                if hasattr(module, 'weight'):
                    print(f"    权重形状: {module.weight.shape}")
                    print(f"    权重设备: {module.weight.device}")
                    print(f"    权重类型: {module.weight.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 Draft Model权重分析和测试工具")
    
    # 权重文件路径
    checkpoint_path = "/Users/zhanghuaxiang/zhx/go_learner/src/SpecForge-1/cache/dataset/Qwen3-8B-eagle3/epoch_9"
    
    # 1. 分析权重结构
    result = analyze_weights_structure(checkpoint_path)
    if result is None:
        print("❌ 权重分析失败，退出")
        return
    
    weights, embedding_keys, non_embedding_keys = result
    
    # 2. 移除embedding并保存
    no_embed_path = remove_embedding_and_save(weights, embedding_keys, non_embedding_keys, checkpoint_path)
    if no_embed_path is None:
        print("❌ 移除embedding失败，退出")
        return
    
    # 3. 测试加载功能
    success = test_loading_with_from_pretrained(no_embed_path)
    
    if success:
        print("\n🎉 所有测试通过！")
        print("✅ 证明：即使预训练权重没有embedding，模型也能正常加载")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main()