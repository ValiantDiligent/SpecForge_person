from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def analyze_model_structure():
    """分析模型结构，展示参数名称"""
    print("="*80)
    print("模型结构分析")
    print("="*80)
    
    state_dict = model.state_dict()
    
    print(f"总参数数量: {len(state_dict)}")
    print("\n前20个参数名称:")
    for i, (name, param) in enumerate(state_dict.items()):
        if i < 20:
            print(f"{i+1:2d}. {name:<50} {param.shape}")
    
    print("\n所有参数名称（按类型分组）:")
    embedding_params = []
    transformer_params = []
    output_params = []
    
    for name, param in state_dict.items():
        if "embed" in name.lower():
            embedding_params.append(name)
        elif "lm_head" in name.lower() or "output" in name.lower():
            output_params.append(name)
        else:
            transformer_params.append(name)
    
    print(f"\nEmbedding层参数 ({len(embedding_params)}个):")
    for name in embedding_params:
        print(f"  - {name}")
    
    print(f"\nTransformer层参数 ({len(transformer_params)}个):")
    for name in transformer_params[:10]:  # 只显示前10个
        print(f"  - {name}")
    if len(transformer_params) > 10:
        print(f"  ... 还有{len(transformer_params)-10}个参数")
    
    print(f"\n输出层参数 ({len(output_params)}个):")
    for name in output_params:
        print(f"  - {name}")
    
    return state_dict

def save_weights_examples(state_dict):
    """演示不同的权重保存方法"""
    print("\n" + "="*80)
    print("权重保存示例")
    print("="*80)
    
    # 示例1: 使用字典推导式排除embedding层
    print("\n1. 排除embedding层的权重（字典推导式）:")
    no_embedding_dict = {
        k: v for k, v in state_dict.items() 
        if "embed" not in k.lower()
    }
    print(f"原始参数数量: {len(state_dict)}")
    print(f"排除embedding后: {len(no_embedding_dict)}")
    print("被排除的参数:")
    excluded = [k for k in state_dict.keys() if "embed" in k.lower()]
    for name in excluded:
        print(f"  - {name}")
    
    # 示例2: 只保存transformer层
    print("\n2. 只保存transformer层（字典推导式）:")
    transformer_only_dict = {
        k: v for k, v in state_dict.items()
        if "layers" in k or "norm" in k
    }
    print(f"只保存transformer层: {len(transformer_only_dict)}")
    
    # 示例3: 重命名参数（类似Eagle3的做法）
    print("\n3. 重命名参数前缀（字典推导式）:")
    # 假设我们要将所有参数加上"my_model."前缀
    renamed_dict = {
        f"my_model.{k}": v for k, v in state_dict.items()
    }
    print(f"重命名后的前5个参数:")
    for i, name in enumerate(renamed_dict.keys()):
        if i < 5:
            print(f"  - {name}")
    
    # 示例4: 条件过滤和重命名组合
    print("\n4. 复杂过滤：排除embedding，重命名transformer层:")
    complex_dict = {
        k.replace("model.", "backbone."): v 
        for k, v in state_dict.items()
        if "embed" not in k.lower() and "lm_head" not in k.lower()
    }
    print(f"复杂过滤后参数数量: {len(complex_dict)}")
    
    # 示例5: 按层级过滤
    print("\n5. 只保存特定层级（如前6层）:")
    specific_layers_dict = {
        k: v for k, v in state_dict.items()
        if any(f"layers.{i}." in k for i in range(6)) or "embed" in k.lower()
    }
    print(f"前6层+embedding参数数量: {len(specific_layers_dict)}")
    
    return no_embedding_dict, transformer_only_dict

def demonstrate_dict_comprehension():
    """演示字典推导式的各种用法"""
    print("\n" + "="*80)
    print("字典推导式详细演示")
    print("="*80)
    
    # 基础示例
    print("\n1. 基础字典推导式:")
    numbers = [1, 2, 3, 4, 5]
    squared = {x: x**2 for x in numbers}
    print(f"原始列表: {numbers}")
    print(f"平方字典: {squared}")
    
    # 条件过滤
    print("\n2. 带条件的字典推导式:")
    even_squared = {x: x**2 for x in numbers if x % 2 == 0}
    print(f"偶数平方: {even_squared}")
    
    # 键值转换
    print("\n3. 键值转换:")
    original = {"a": 1, "b": 2, "c": 3}
    swapped = {v: k for k, v in original.items()}
    print(f"原始字典: {original}")
    print(f"键值互换: {swapped}")
    
    # 字符串处理
    print("\n4. 字符串处理（类似Eagle3的做法）:")
    model_params = {
        "model.embed_tokens.weight": "tensor1",
        "model.layers.0.weight": "tensor2", 
        "model.layers.1.weight": "tensor3",
        "other.param": "tensor4"
    }
    
    # 提取model层参数并重命名
    model_only = {
        k.replace("model.", ""): v 
        for k, v in model_params.items()
        if k.startswith("model.")
    }
    print(f"原始参数: {model_params}")
    print(f"提取model层: {model_only}")

def save_model_weights():
    """实际保存模型权重的示例"""
    print("\n" + "="*80)
    print("实际保存权重示例")
    print("="*80)
    
    state_dict = model.state_dict()
    
    # 方法1: 保存完整模型（不推荐用于大模型）
    print("\n1. 保存完整模型:")
    # torch.save(model, "complete_model.pth")  # 注释掉避免实际保存
    print("torch.save(model, 'complete_model.pth')  # 保存整个模型对象")
    
    # 方法2: 保存state_dict（推荐）
    print("\n2. 保存完整state_dict:")
    # torch.save(state_dict, "model_weights.pth")
    print("torch.save(state_dict, 'model_weights.pth')  # 只保存权重")
    
    # 方法3: 保存过滤后的权重
    print("\n3. 保存过滤后的权重:")
    no_embedding_dict = {
        k: v for k, v in state_dict.items() 
        if "embed" not in k.lower()
    }
    # torch.save(no_embedding_dict, "no_embedding_weights.pth")
    print("torch.save(no_embedding_dict, 'no_embedding_weights.pth')")
    
    # 方法4: 使用HuggingFace的save_pretrained
    print("\n4. 使用HuggingFace方法:")
    print("model.save_pretrained('./saved_model')  # 保存为HF格式")
    
    # 方法5: 自定义保存函数
    print("\n5. 自定义保存函数:")
    def save_filtered_weights(model, save_path, exclude_patterns=None):
        """保存过滤后的模型权重"""
        if exclude_patterns is None:
            exclude_patterns = ["embed"]
        
        state_dict = model.state_dict()
        filtered_dict = {}
        
        for name, param in state_dict.items():
            should_exclude = any(pattern in name.lower() for pattern in exclude_patterns)
            if not should_exclude:
                filtered_dict[name] = param
        
        # torch.save(filtered_dict, save_path)
        print(f"保存{len(filtered_dict)}个参数到 {save_path}")
        return filtered_dict
    
    # 使用自定义函数
    filtered_weights = save_filtered_weights(
        model, 
        "filtered_weights.pth", 
        exclude_patterns=["embed", "lm_head"]
    )
    print(f"过滤后参数数量: {len(filtered_weights)}")

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
        max_new_tokens=512,  # 减少token数量
        do_sample=True,
        temperature=0.7
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # 输出完整内容
    full_content = tokenizer.decode(output_ids, skip_special_tokens=False)
    
    print("\n生成的回答:")
    print(full_content)
    
    return full_content

if __name__ == "__main__":
    print("开始模型分析和权重保存演示")
    
    # 1. 分析模型结构
    state_dict = analyze_model_structure()
    
    # 2. 演示字典推导式
    demonstrate_dict_comprehension()
    
    # 3. 演示权重保存方法
    save_weights_examples(state_dict)
    
    # 4. 演示实际保存
    save_model_weights()
    
    # 5. 简单测试模型
    print("\n" + "="*80)
    print("模型功能测试")
    print("="*80)
    
    test_prompt("什么是字典推导式？", "字典推导式测试")
    
    print(f"\n{'='*80}")
    print("演示完成!")
    print("="*80)