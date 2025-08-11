#!/usr/bin/env python3
"""
分析和处理draft model权重的脚本
功能：
1. 查看权重结构
2. 移除embedding层
3. 测试保存和加载
4. 支持safetensor格式
"""

import os
import torch
import json
from pathlib import Path
from collections import OrderedDict
import sys
import argparse
from specforge import (
    AutoEagle3DraftModel,
)

# 尝试导入safetensors
try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safe_save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    print("⚠️ safetensors not available, will use pytorch format")
    SAFETENSORS_AVAILABLE = False


def detect_model_format(checkpoint_dir):
    """
    检测模型文件格式
    
    Returns:
        str: 'safetensors', 'pytorch', or 'unknown'
    """
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensor_path):
        return 'safetensors'
    elif os.path.exists(pytorch_path):
        return 'pytorch'
    else:
        return 'unknown'


def load_state_dict_from_checkpoint(checkpoint_dir):
    """
    从检查点目录加载state dict，自动检测格式
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        tuple: (state_dict, format_type)
    """
    format_type = detect_model_format(checkpoint_dir)
    
    if format_type == 'safetensors' and SAFETENSORS_AVAILABLE:
        safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
        print(f"📂 加载safetensor格式: {safetensor_path}")
        
        state_dict = {}
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        return state_dict, 'safetensors'
        
    elif format_type == 'pytorch':
        pytorch_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        print(f"📂 加载pytorch格式: {pytorch_path}")
        
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        return state_dict, 'pytorch'
        
    else:
        raise ValueError(f"无法找到支持的模型文件格式在 {checkpoint_dir}")


def save_state_dict_to_checkpoint(state_dict, output_dir, format_type='safetensors'):
    """
    保存state dict到检查点目录
    
    Args:
        state_dict: 要保存的state dict
        output_dir: 输出目录
        format_type: 保存格式 ('safetensors' 或 'pytorch')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type == 'safetensors' and SAFETENSORS_AVAILABLE:
        safetensor_path = os.path.join(output_dir, "model.safetensors")
        print(f"💾 保存为safetensor格式: {safetensor_path}")
        
        # 确保所有tensor都在CPU上
        cpu_state_dict = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in state_dict.items()}
        safe_save_file(cpu_state_dict, safetensor_path)
        
        return safetensor_path
        
    else:
        pytorch_path = os.path.join(output_dir, "pytorch_model.bin")
        print(f"💾 保存为pytorch格式: {pytorch_path}")
        
        torch.save(state_dict, pytorch_path)
        return pytorch_path


def print_model_info(model, title="模型信息"):
    """打印模型详细信息"""
    print(f"\n{'='*20} {title} {'='*20}")
    print("模型结构：")
    print(model)
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print("=" * 70)
    print("所有参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {tuple(param.shape)} - {'trainable' if param.requires_grad else 'frozen'}")
    print("=" * 70)


def analyze_state_dict(state_dict, title="State Dict分析"):
    """分析state dict结构"""
    print(f"\n{'='*20} {title} {'='*20}")
    total_params = 0
    embed_params = 0
    
    for key, tensor in state_dict.items():
        param_count = tensor.numel()
        total_params += param_count
        
        if "embed" in key.lower():
            embed_params += param_count
            print(f"🔍 EMBEDDING: {key}: {tuple(tensor.shape)} ({param_count:,} params)")
        else:
            print(f"   {key}: {tuple(tensor.shape)} ({param_count:,} params)")
    
    print(f"\n📊 总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"📊 Embedding参数量: {embed_params:,} ({embed_params/1e6:.2f}M)")
    print(f"📊 非Embedding参数量: {(total_params-embed_params):,} ({(total_params-embed_params)/1e6:.2f}M)")
    print("=" * 70)


def remove_embedding_from_state_dict(state_dict, embedding_keys=None):
    """
    从state dict中移除embedding相关的权重
    
    Args:
        state_dict: 原始state dict
        embedding_keys: 要移除的embedding key列表，如果为None则自动检测
    
    Returns:
        新的state dict（不包含embedding）
    """
    if embedding_keys is None:
        # 自动检测embedding相关的key
        embedding_keys = [
            "embed_tokens.weight",
            "model.embed_tokens.weight", 
            "embeddings.word_embeddings.weight",
            "word_embeddings.weight"
        ]
    
    new_state_dict = OrderedDict()
    removed_keys = []
    
    for key, value in state_dict.items():
        # 检查是否是embedding相关的key
        should_remove = False
        for embed_key in embedding_keys:
            if embed_key in key or "embed" in key.lower():
                should_remove = True
                removed_keys.append(key)
                break
        
        if not should_remove:
            new_state_dict[key] = value
    
    print(f"\n🗑️ 移除的embedding层:")
    for key in removed_keys:
        print(f"   - {key}")
    
    return new_state_dict, removed_keys


def copy_other_files(input_dir, output_dir):
    """
    复制其他必要的文件（config.json, training_state.pt等）
    """
    files_to_copy = [
        "config.json",
        "training_state.pt",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]
    
    copied_files = []
    for filename in files_to_copy:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if os.path.exists(src_path):
            import shutil
            shutil.copy2(src_path, dst_path)
            copied_files.append(filename)
            print(f"📋 复制文件: {filename}")
    
    return copied_files


def process_checkpoint_directly(input_dir, output_dir, embedding_keys=None, preserve_format=True):
    """
    直接处理检查点文件，移除embedding层
    
    Args:
        input_dir: 输入检查点目录
        output_dir: 输出目录
        embedding_keys: 要移除的embedding key列表
        preserve_format: 是否保持原始格式
    """
    print(f"\n🔄 直接处理检查点文件")
    
    # 检测输入格式
    input_format = detect_model_format(input_dir)
    print(f"📂 检测到输入格式: {input_format}")
    
    if input_format == 'unknown':
        raise ValueError(f"无法识别的模型格式在 {input_dir}")
    
    # 加载state dict
    state_dict, detected_format = load_state_dict_from_checkpoint(input_dir)
    print(f"📋 加载了 {len(state_dict)} 个参数")
    
    # 分析原始state dict
    analyze_state_dict(state_dict, "原始State Dict")
    
    # 移除embedding
    new_state_dict, removed_keys = remove_embedding_from_state_dict(
        state_dict, embedding_keys
    )
    
    # 分析新的state dict
    analyze_state_dict(new_state_dict, "移除Embedding后的State Dict")
    
    # 确定输出格式
    output_format = detected_format if preserve_format else 'safetensors'
    if not SAFETENSORS_AVAILABLE and output_format == 'safetensors':
        output_format = 'pytorch'
        print("⚠️ safetensors不可用，使用pytorch格式")
    
    # 保存新的state dict
    saved_path = save_state_dict_to_checkpoint(new_state_dict, output_dir, output_format)
    print(f"✅ 模型权重已保存: {saved_path}")
    
    # 复制其他文件
    copied_files = copy_other_files(input_dir, output_dir)
    
    # 保存移除信息
    removed_info = {
        "removed_keys": removed_keys,
        "original_param_count": len(state_dict),
        "new_param_count": len(new_state_dict),
        "removed_param_count": len(removed_keys),
        "input_format": detected_format,
        "output_format": output_format,
        "copied_files": copied_files
    }
    
    info_path = os.path.join(output_dir, "removed_embedding_info.json")
    with open(info_path, 'w') as f:
        json.dump(removed_info, f, indent=2)
    print(f"✅ 移除信息已保存: {info_path}")
    
    return new_state_dict, removed_keys


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="移除draft model的embedding层")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="输入模型检查点目录")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录（保存移除embedding后的模型）")
    parser.add_argument("--embedding-keys", type=str, nargs="+", 
                       default=["embed_tokens.weight"],
                       help="要移除的embedding key列表")
    parser.add_argument("--analyze-only", action="store_true",
                       help="只分析模型，不保存")
    parser.add_argument("--direct-process", action="store_true",
                       help="直接处理检查点文件（不通过模型加载）")
    parser.add_argument("--output-format", type=str, choices=['safetensors', 'pytorch', 'auto'],
                       default='auto', help="输出格式")
    
    args = parser.parse_args()
    
    print("🚀 Draft Model Embedding移除工具")
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎯 要移除的embedding keys: {args.embedding_keys}")
    print(f"🔧 Safetensors支持: {'✅' if SAFETENSORS_AVAILABLE else '❌'}")
    
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    try:
        if args.analyze_only:
            # 只分析模型
            print(f"\n🔍 分析模式：只查看模型结构")
            
            if args.direct_process:
                # 直接分析文件
                state_dict, format_type = load_state_dict_from_checkpoint(args.input_dir)
                analyze_state_dict(state_dict, f"State Dict分析 ({format_type})")
            else:
                # 通过模型加载分析
                draft_model = AutoEagle3DraftModel.from_pretrained(args.input_dir).to(torch.bfloat16)
                print_model_info(draft_model, "模型分析")
                analyze_state_dict(draft_model.state_dict(), "State Dict分析")
        else:
            # 处理和保存
            if args.direct_process:
                # 直接处理文件
                preserve_format = (args.output_format == 'auto')
                new_state_dict, removed_keys = process_checkpoint_directly(
                    args.input_dir, args.output_dir, args.embedding_keys, preserve_format
                )
            else:
                # 通过模型加载处理
                print("⚠️ 使用模型加载方式，将保存为pytorch格式")
                draft_model = AutoEagle3DraftModel.from_pretrained(args.input_dir).to(torch.bfloat16)
                print_model_info(draft_model, "原始模型")
                analyze_state_dict(draft_model.state_dict(), "原始State Dict")
                
                # 移除embedding并保存
                original_state_dict = draft_model.state_dict()
                new_state_dict, removed_keys = remove_embedding_from_state_dict(
                    original_state_dict, args.embedding_keys
                )
                
                # 保存
                os.makedirs(args.output_dir, exist_ok=True)
                model_path = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(new_state_dict, model_path)
                print(f"✅ 模型权重已保存: {model_path}")
                
                # 复制其他文件
                copy_other_files(args.input_dir, args.output_dir)
        
        print(f"\n✨ 处理完成!")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()