from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def test_prompt(prompt, description=""):
    """测试单个prompt并显示完整输出"""
    print(f"\n{'='*60}")
    if description:
        print(f"测试: {description}")
    print(f"Prompt: {prompt}")
    print('='*60)
    
    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # 启用思考模式
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,  # 降低token数量便于测试
        do_sample=True,
        temperature=0.7
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # 修改：不区分思考和非思考部分，直接输出完整内容
    # 保留所有特殊token，包括<think>和</think>标签
    full_content = tokenizer.decode(output_ids, skip_special_tokens=False)
    
    # print("完整输出（包含所有特殊子树）:")
    # print(full_content)  # 这会让\n显示为真正的换行
    
    print("\n原始字符串表示（可见\\n字符）:")
    print(repr(full_content))  # 这会显示\n字符而不是换行
    
    # 分析输出中的特殊标签
    special_tags = re.findall(r'<[^>]+>', full_content)
    if special_tags:
        print(f"\n检测到的特殊标签: {list(set(special_tags))}")
    
    # 显示原始token信息
    print(f"\n输出Token数量: {len(output_ids)}")
    print(f"前20个Token IDs: {output_ids[:20]}")
    
    # 检查是否包含thinking相关的特殊token
    think_start_id = 151667  # <think> token ID (可能需要根据实际调整)
    think_end_id = 151668    # </think> token ID
    
    has_think_start = think_start_id in output_ids
    has_think_end = think_end_id in output_ids
    
    print(f"包含<think>标签: {has_think_start}")
    print(f"包含</think>标签: {has_think_end}")
    
    return full_content

# 测试多个不同类型的prompt
test_prompts = [
    ("Give me a short introduction to large language model.", "原始测试"),
    ("解释一下什么是机器学习，请详细思考。", "中文思考测试"),
    ("Solve this step by step: What is 25 * 17?", "数学计算测试"),
    ("分析一下人工智能的发展前景，需要深入思考。", "复杂分析测试"),
    ("Write a Python function to calculate fibonacci numbers.", "编程测试")
]

print("开始测试 - 输出将包含所有特殊子树，不区分思考和非思考部分")

for prompt, description in test_prompts:
    try:
        result = test_prompt(prompt, description)
    except Exception as e:
        print(f"测试失败: {e}")
        continue

print(f"\n{'='*60}")
print("所有测试完成!")
print("注意：输出保留了完整的特殊标签结构，包括<think>...</think>等特殊子树")