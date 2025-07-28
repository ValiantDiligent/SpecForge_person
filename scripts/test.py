import requests
import json
url = f"http://localhost:30000/v1/chat/completions"

path = "/root/SpecForge/cache/dataset/test725.jsonl"
NUMS = 1000  
num = 0    
with open(path, "r") as f:

    for line in f:
        line = line.strip()
        if not line:
            continue
        if num >= NUMS:
            print(f"已达到最大处理数量 {NUMS}，停止处理")
            break
        num += 1
        json_data = json.loads(line)
        conversations_list = json_data['conversations']
        if(len(conversations_list) >= 3):
            conversations_list.pop()
        
        data_dict = {
            "model": "Qwen/Qwen3-8B",
            "messages": conversations_list 
        }
        response = requests.post(url, json=data_dict)
        # data_s = json.dumps(data_dict,ensure_ascii=False)
# print(response.json())