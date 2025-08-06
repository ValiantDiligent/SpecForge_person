from transformers import AutoTokenizer
import json
import re

# 初始化分词器 - 使用 Qwen3
model_name = "Qwen/Qwen3-8B"  # 或者 "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading tokenizer for {model_name}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Failed to load {model_name}, trying backup model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Successfully loaded: {model_name}")

def display_with_newlines(text, title):
    """显示文本，保留换行符的可见性"""
    print(f"\n{title}:")
    print("=" * 60)
    print("Raw representation (visible \\n):")
    print(repr(text))
    print("\nFormatted text:")
    print(text)
    print("=" * 60)

def analyze_tokens(text, title):
    """分析token，包括特殊token"""
    print(f"\n{title} Token Analysis:")
    print("-" * 40)
    
    # 编码（保留特殊token）
    tokens_with_special = tokenizer.encode(text, add_special_tokens=False)
    tokens_regular = tokenizer.encode(text, add_special_tokens=True)
    
    print(f"Token count (no special): {len(tokens_with_special)}")
    print(f"Token count (with special): {len(tokens_regular)}")
    print(f"First 15 token IDs: {tokens_with_special[:15]}")
    
    # 检查思考相关的特殊token
    think_start_ids = [151667, 151664]  # 可能的<think> token IDs
    think_end_ids = [151668, 151665]    # 可能的</think> token IDs
    
    has_think_start = any(tid in tokens_with_special for tid in think_start_ids)
    has_think_end = any(tid in tokens_with_special for tid in think_end_ids)
    
    print(f"Contains <think> tokens: {has_think_start}")
    print(f"Contains </think> tokens: {has_think_end}")
    
    # 解码验证
    decoded_with_special = tokenizer.decode(tokens_with_special, skip_special_tokens=False)
    decoded_clean = tokenizer.decode(tokens_with_special, skip_special_tokens=True)
    
    print("Decoded (with special tokens):")
    print(repr(decoded_with_special))
    
    # 检查特殊标签
    special_tags = re.findall(r'<[^>]+>', decoded_with_special)
    if special_tags:
        print(f"Special tags found: {list(set(special_tags))}")
    else:
        print("No special tags found")
    
    return tokens_with_special, decoded_with_special

def test_chat_template(messages, description, test_thinking=True):
    """测试聊天模板，包括思考和非思考模式"""
    print(f"\n\n{'#' * 80}")
    print(f"# {description}")
    print(f"{'#' * 80}")
    
    print("Input Messages:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    
    # 测试不同的思考模式
    thinking_modes = []
    if test_thinking:
        thinking_modes = [
            {"enable_thinking": False, "name": "非思考模式"},
            {"enable_thinking": True, "name": "思考模式"}
        ]
    else:
        thinking_modes = [{"enable_thinking": False, "name": "标准模式"}]
    
    for mode in thinking_modes:
        print(f"\n{'-' * 50}")
        print(f"模式: {mode['name']}")
        print(f"{'-' * 50}")
        
        try:
            # 应用聊天模板
            if 'enable_thinking' in mode:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=mode['enable_thinking']
                )
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            display_with_newlines(formatted_prompt, f"Chat Template Output ({mode['name']})")
            analyze_tokens(formatted_prompt, mode['name'])
            
        except Exception as e:
            print(f"Error in {mode['name']}: {e}")
            # 尝试不带enable_thinking参数
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                display_with_newlines(formatted_prompt, f"Fallback Template Output")
                analyze_tokens(formatted_prompt, "Fallback")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")

# 显示聊天模板
print(f"\n{'=' * 80}")
print(f"Chat Template for {model_name}:")
print(f"{'=' * 80}")
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print(tokenizer.chat_template)
else:
    print("No chat template found")

# 测试用例1: 简单问候
test_chat_template(
    [{"role": "user", "content": "你好！"}],
    "测试1: 简单用户问候"
)

# 测试用例2: 包含换行符的复杂内容
test_chat_template(
    [{"role": "user", "content": "请解释机器学习：\n1. 什么是机器学习？\n2. 主要应用领域\n3. 未来发展趋势\n\n请详细回答每个问题。"}],
    "测试2: 包含多个换行符的复杂问题"
)

# 测试用例3: 带系统提示词的简单对话
test_chat_template(
    [
        {"role": "system", "content": "你是一个专业的AI助手，请保持礼貌和准确。"},
        {"role": "user", "content": "什么是量子计算？"}
    ],
    "测试3: 带系统提示词的对话"
)

# 测试用例4: 复杂系统提示词
test_chat_template(
    [
        {"role": "system", "content": "你是一个Python编程专家。\n规则：\n1. 提供详细的代码解释\n2. 包含实际可运行的示例\n3. 说明最佳实践\n4. 指出常见错误\n\n请始终遵循这些规则。"},
        {"role": "user", "content": "如何实现一个高效的快速排序算法？"}
    ],
    "测试4: 复杂多行系统提示词"
)

# 测试用例5: 多轮对话
test_chat_template(
    [
        {"role": "user", "content": "什么是深度学习？"},
        {"role": "assistant", "content": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的复杂模式。"},
        {"role": "user", "content": "它与传统机器学习有什么区别？\n请从以下几个方面对比：\n- 数据需求\n- 计算复杂度\n- 应用场景"}
    ],
    "测试5: 多轮对话与换行符"
)

# 测试用例6: 复杂系统提示词 + 多轮对话
test_chat_template(
    [
        {"role": "system", "content": "你是一个哲学教授。\n请注意：\n- 深入分析每个概念\n- 提供历史背景\n- 举出具体例子\n- 引用相关哲学家的观点"},
        {"role": "user", "content": "什么是自由意志？"},
        {"role": "assistant", "content": "自由意志是哲学中的一个核心概念...\n\n从历史角度看，这个问题可以追溯到古希腊时期。"},
        {"role": "user", "content": "如果没有自由意志，道德责任还存在吗？\n这个问题很复杂，请深入思考。"}
    ],
    "测试6: 复杂系统提示词 + 多轮对话"
)

# 测试用例7: 需要深度思考的数学问题（专门测试思考模式）
test_chat_template(
    [
        {"role": "user", "content": "请解决这个复杂的数学问题：\n\n一个正八面体的体积公式是什么？\n如果边长为a，请：\n1. 推导体积公式\n2. 计算当a=6时的体积\n3. 说明推导过程\n\n这需要仔细思考几何关系。"}
    ],
    "测试7: 复杂数学推导问题（测试思考模式）"
)

# 测试用例8: 编程问题（容易触发思考）
test_chat_template(
    [
        {"role": "user", "content": "设计一个算法来解决以下问题：\n\n给定一个字符串，找出其中最长的回文子串。\n要求：\n- 时间复杂度尽可能低\n- 空间复杂度也要考虑\n- 处理边界情况\n- 提供完整的Python实现\n\n请详细分析不同算法的优缺点。"}
    ],
    "测试8: 算法设计问题（测试思考模式）"
)

# 测试用例9: 哲学思辨问题
test_chat_template(
    [
        {"role": "system", "content": "你是一个深度思考者，擅长哲学分析。"},
        {"role": "user", "content": "如果人工智能达到了人类的智能水平，\n甚至超越了人类，\n那么：\n\n1. 它们是否应该拥有权利？\n2. 人类如何定义自己的价值？\n3. 这种情况下的伦理框架应该是什么？\n\n这是一个需要深入思考的复杂问题。"}
    ],
    "测试9: 复杂哲学思辨问题（重点测试思考模式）"
)

print(f"\n\n{'=' * 80}")
print("🎉 所有测试完成！")
print("=" * 80)
print("观察要点：")
print("1. 💭 换行符在repr()中显示为\\n，在格式化文本中正常显示")
print("2. 🔄 思考模式vs非思考模式的模板差异")
print("3. 🏷️  特殊token (如<think>, </think>)的检测")
print("4. 🤖 系统提示词的处理方式")
print("5. 💬 多轮对话的模板结构")
print("6. 🧠 复杂问题是否触发思考模式")
print("7. 📊 Token数量和ID的分析")
print("=" * 80)