from datasets import load_dataset

# 测试加载你的 JSONL 文件
try:
    dataset = load_dataset("json", data_files="/Users/zhanghuaxiang/zhx/SpecSam/DATA/train-eagle-wq-0802-qwen3-32b-100000.json")["train"]
    print(f"✅ 成功加载 {len(dataset)} 条数据")
    print(f"数据列: {dataset.column_names}")
    
    if len(dataset) > 0:
        print(f"第一条数据: {dataset[0]}")
        
        # 检查是否有 conversations 字段
        if "conversations" in dataset[0]:
            print("✅ 包含 conversations 字段")
            conversations = dataset[0]["conversations"]
            print(f"对话轮数: {len(conversations)}")
            for i, turn in enumerate(conversations):
                print(f"  轮次 {i+1}: {turn.get('role', 'unknown')} - {turn.get('content', '')[:50]}...")
        else:
            print("❌ 缺少 conversations 字段")
            print(f"当前字段: {list(dataset[0].keys())}")
            
except Exception as e:
    print(f"❌ 加载失败: {e}")