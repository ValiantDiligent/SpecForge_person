from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"--- Chat Template for {model_name} ---")
print(tokenizer.chat_template)


from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


chat_example = [
   # {"role": "system", "content": "You are a helpful assistant for travel planning."},
    {"role": "user", "content": "I want to go to Paris, what should I see?"},
]

# 应用聊天模板
formatted_prompt = tokenizer.apply_chat_template(
    chat_example, 
    tokenize=False, # 设置为 False 以返回字符串，而不是 token IDs
    add_generation_prompt=True # 添加助手的起始提示，让模型知道该它说话了
)

print(f"--- Formatted Prompt for {model_name} ---")
print(formatted_prompt)





##
meta-llama/Llama-3.1-8B-Instruct 
model_name = "meta-llama/Llama-3.1-8B-Instruct "
##