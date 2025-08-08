#!/usr/bin/env python3
"""
分析和处理draft model权重的脚本
功能：
1. 查看权重结构
2. 移除embedding层
3. 测试保存和加载
"""

import os
import torch
import json
from pathlib import Path
from collections import OrderedDict
import sys
from specforge import (
    AutoEagle3DraftModel,
)


def print_model_info(model):
    print("模型结构：")
    print(model)
    print("\n模型总参数量: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    print("=" * 70)
    print("所有参数:")
    for name, param in model.named_parameters():
        print(f"{name}: {tuple(param.shape)} - {'trainable' if param.requires_grad else 'frozen'}")
    print("=" * 70)

def main():
    """主函数"""
    print("🚀 Draft Model权重分析和处理工具")
    
    # 路径配置
    checkpoint_dir = "/Users/zhanghuaxiang/zhx/go_learner/src/SpecForge-1/cache/dataset/Qwen3-8B-eagle3/epoch_9"
    draft_model = (
            AutoEagle3DraftModel.from_pretrained(checkpoint_dir)
            .to(torch.bfloat16)
        )
    print('='*80)
    print_model_info(draft_model)
    print(f"\n✨ 分析完成!")


if __name__ == "__main__":
    main()