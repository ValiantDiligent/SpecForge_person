#!/usr/bin/env python3
"""
测试字符串在 Qwen3 中的 tokenization
"""

from transformers import AutoTokenizer

def test_tokenization():
    # 使用 Qwen3 tokenizer
    model_name = "Qwen/Qwen3-0.6B"  # 或者你本地的 Qwen3 模型路径
    print(f"加载 tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"加载失败，尝试备用模型: {e}")
        # 备用模型
        model_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 要测试的字符串
    test_string = '<think>\n\n</think>\n\n否<|im_end|>'
    
    print("=" * 60)
    print(f"测试字符串: {repr(test_string)}")
    print(f"字符串长度: {len(test_string)} 个字符")
    print("=" * 60)
    
    # 进行 tokenization（不添加特殊 token）
    input_ids = tokenizer.encode(test_string, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    print(f"\n📊 Token 统计:")
    print(f"Token 数量: {len(input_ids)} 个")
    print(f"Token 详情:")
    
    for i, (token, token_id) in enumerate(zip(tokens, input_ids)):
        # 显示可读的 token 表示
        display_token = token.replace('\n', '\\n').replace(' ', '▁')
        print(f"  {i+1:2d}. '{display_token}' (ID: {token_id})")
    
    # 验证解码
    decoded = tokenizer.decode(input_ids)
    print(f"\n🔍 解码验证:")
    print(f"解码结果: {repr(decoded)}")
    print(f"解码匹配: {'✅' if decoded == test_string else '❌'}")
    
    # 分析各个组成部分
    print(f"\n🔧 组件分析:")
    components = [
        '<think>',  # 开始标签
        '\n\n',     # 双换行
        '</think>', # 结束标签
        '\n\n',       # 单换行
        '否',       # 中文字符
        '<|im_end|>',        # 结束
        '<|im_end|>\n' 
    ]
    
    total_component_tokens = 0
    for comp in components:
        comp_ids = tokenizer.encode(comp, add_special_tokens=False)
        comp_tokens = tokenizer.convert_ids_to_tokens(comp_ids)
        display_comp = comp.replace('\n', '\\n')
        display_tokens = [t.replace('\n', '\\n').replace(' ', '▁') for t in comp_tokens]
        print(f"  '{display_comp}' -> {len(comp_ids)} tokens: {display_tokens}")
        total_component_tokens += len(comp_ids)
    
    print(f"\n组件总 token 数: {total_component_tokens}")
    print(f"整体 token 数: {len(input_ids)}")
    print(f"差异: {len(input_ids) - total_component_tokens} (可能由于上下文效应)")
    
    # 测试不同的变体
    print(f"\n🧪 变体测试:")
    variants = [
        '<think>\n\n</think>\n\n否',  # 无引号版本
        '<think></think>否',        # 无换行版本
        '<think>\n\n</think>\n\n否<|im_end|>',  # 包含结束标记
        '<think>\n\n</think>\n\n    ',                      # 只有中文字符
        '<think>\n\n</think>\n\n',     # 只有标签部分
    ]
    
    for variant in variants:
        var_ids = tokenizer.encode(variant, add_special_tokens=False)
        display_variant = variant.replace('\n', '\\n')
        print(f"  '{display_variant}' -> {len(var_ids)} tokens")

if __name__ == "__main__":
    test_tokenization()