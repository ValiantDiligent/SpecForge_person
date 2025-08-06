import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --------------------------------------------------------------------------
# 1. 配置和加载模型
# --------------------------------------------------------------------------

# 定义模型 ID
# 注意：我们使用的是 Qwen2-7B-Instruct，因为 Qwen3-8B 并非官方发布的模型名称
model_name = "Qwen/Qwen3-0.6B"

print(f"正在加载模型: {model_name}")

# 加载分词器 (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型
# - torch_dtype="auto": 自动选择最优的数据类型 (如 bfloat16) 以加速并节省显存
# - device_map="auto": 自动将模型加载到可用的硬件上 (如 GPU)，需要 `accelerate` 库
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print("模型和分词器加载完成！")

# --------------------------------------------------------------------------
# 2. 准备对话内容
# --------------------------------------------------------------------------

# Qwen2 的聊天模板需要一个角色列表 (system, user, assistant)
# 我们可以提供一个 system prompt 来设定模型的角色和行为
messages = [{'role': 'system', 'content': '你是一位专业的电商售后运营智能助手，请根据买家的退款申请信息（包括商品类目、退款类型和退款描述），判断其真实退款原因，并返回**结构化标签 + 置信度评分**。\n\n---\n\n# 输入信息结构：\n- 商品类目：如女装、生鲜、绿植、数码、虚拟商品、本地生活等\n- 退款类型：仅退款 / 退货退款\n- 退款描述：买家自由填写的文本内容\n\n---\n\n# 任务目标：\n请从下方19个标准退款原因标签中，推理买家最可能的退款意图，并输出：\n1. **item**：推理出的最可能的结构化退款原因标签；\n2. **reason**：给出提取该退款原因的理由；\n3. **confidenceIndex**：该结果的置信度分数，范围为0到1，保留两位小数；\n4. 如推理无法确定，请返回“其他”，并适当降低confidence值（例如0.3以下）。\n\n---\n\n# 标签说明与示例：\n\n（以下每项都含定义+示例，请大模型理解标签语义）\n\n1. **物流问题**：物流慢、异常、快递服务差  \n   - 示例：快递迟迟不到 / 物流没人联系我  \n2. **做工差/有瑕疵**：线头、掉漆、开胶、裁剪不整齐  \n   - 示例：鞋子边缘胶开了 / 毛衣有线头  \n3. **质量问题/功效问题**：功能损坏、无法使用、效果无效  \n   - 示例：吹风机加热失败 / 眼霜用了没感觉  \n4. **脏污/破损**：商品或包装有污损、裂痕  \n   - 示例：包裹外面脏了 / 杯子碎成几块  \n5. **生鲜绿植变质/死亡/缺斤少两**：腐烂、萎蔫、重量不足  \n   - 示例：葡萄发霉了 / 绿植一半叶子黑了  \n6. **商品与宣传不符**：颜色、大小、功能与页面描述不一致  \n   - 示例：页面写的是白色，发来是灰色  \n7. **过敏**：使用商品后产生皮肤或身体不适  \n   - 示例：敷了面膜皮肤发红  \n8. **盗版/假冒商品**：收到假货、盗版、非官方商品  \n   - 示例：耳机不是官网正品 / 图书是盗印的  \n9. **变质/临期/三无等食安**：过期、三无食品、标签不清  \n   - 示例：收到牛奶已过保质期  \n10. **虚拟未收到货**：虚拟商品未到账或无法使用  \n    - 示例：话费没到账 / 卡券无法兑换  \n11. **其他**：原因不清或不符合以上任何标签  \n    - 示例：不太满意但说不上来原因  \n12. **跨境专属**：跨境订单特有问题，如清关、版本、发票  \n    - 示例：清关慢 / 是日版无中文说明  \n13. **七天无理由退货**：不喜欢、不适合，但商品完好  \n    - 示例：拿到实物觉得样式不适合  \n14. **本地生活**：到店/本地服务类未履约  \n    - 示例：到店券去不了 / 门店关闭  \n15. **未按约定时间发货**：超出承诺发货时间仍未发货  \n    - 示例：预售期已过还没发货  \n16. **商家缺货**：下单后提示无货，无法发货  \n    - 示例：商家让我退款说没货  \n17. **空包/少漏错**：空包、漏发、发错商品  \n    - 示例：只收到一件衣服 / 收错型号  \n18. **不想要、不喜欢、拍多、拍错**：主观退单原因  \n    - 示例：颜色不喜欢 / 拍多了想退  \n19. **商家发货慢**：未超时但发货不及时引发不满  \n    - 示例：拍下后两天才发货\n\n---\n\n# 输出要求：\n\n请根据输入的商品类目、退款类型和退款描述，综合判断并返回如下格式的结构化结果,输出应为JSON格式：\n- "item": 用户退款原因\n- "confidenceIndex": 识别置信度，从0到1，代表对识别结果的信心程度，越大信心越足\n- "reason": 提取该退款原因的理由\n\n# 示例\n\n**输入示例**\n- 退款类型: "仅退款"\n- 商品类目: "鞋子"\n- 退款描述: "拍错码数"\n\n**输出示例**\n{\n    "item": "不想要、不喜欢、拍多、拍错",\n    "confidenceIndex":0.92,\n    "reason": "XXX"\n}\n\n# 任务流程：\n\n1. 阅读买家退款描述，结合商品类目与退款类型，推理其真实退款原因；\n2. 在上述退款原因标签中，选择**最合适的一项**；\n3. 如果用户描述模糊或不清晰，选择“其他”；\n\n/no_think'}, {'role': 'user', 'content': '- 退款类型: 退货退款\n- 商品类目: 抹胸\n- 退款描述: 不合适大小'}]

# --------------------------------------------------------------------------
# 方法一：使用 model.generate() (更底层的控制)
# --------------------------------------------------------------------------
print("\n--- 方法一: 使用 model.generate() ---")

# 1. 应用聊天模板：将对话列表转换为模型可以理解的输入格式
#    这是至关重要的一步，确保遵循了模型的特定格式
model_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True, # 添加提示，让模型知道该它说话了
    return_tensors="pt"
).to(model.device) # 将输入数据移动到模型所在的设备 (GPU/CPU)

print("\n========== Model Inputs  ==========")
print(model_inputs)

formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print("========== Formatted Prompt String ==========")
print(formatted_prompt)



# 2. 生成回复
#    max_new_tokens 控制生成内容的最大长度
#    eos_token_id 是结束符的ID，模型生成到这里可以提前停止
#    注意: Qwen2 可能有多个结束符，tokenizer.eos_token_id 会处理好
outputs = model.generate(
    model_inputs,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id
)

# 3. 解码输出
#    我们需要从返回的完整序列中，只解码出新生成的部分
response_ids = outputs[0][model_inputs.shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
print("========== 模型的回复(skip special) ==========")
print(response_text)

response_text_2 = tokenizer.decode(response_ids, skip_special_tokens=False)
print("========== 模型的回复(no skip special) ==========")
print(response_text_2)