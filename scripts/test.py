import requests
import json
import time
url = f"http://localhost:30000/v1/chat/completions"

path = "/root/SpecForge/cache/dataset/test725.jsonl"
NUMS = 1000  
num = 0    
start_time = time.time()
with open(path, "r") as f:

    for line in f:
        line = line.strip()
        if not line:
            continue
        if success_count >= NUMS:
            print(f"已达到最大处理数量 {NUMS}，停止处理")
            break
        num += 1
        json_data = json.loads(line)
        conversations_list = json_data['conversations']
        if(len(conversations_list) >= 3):
            conversations_list.pop()
        
        data_dict = {
            "model": "Qwen/Qwen3-8B",
            "messages": conversations_list,
            "temperature": 0
        }
        try:
            response = requests.post(url, json=data_dict)
            if response.status_code == 200:
                success_count += 1
            else:
                failed_count += 1
                print(f"请求失败，状态码: {response.status_code}")
        except Exception as e:
            failed_count += 1
            print(f"请求异常: {e}")
        # data_s = json.dumps(data_dict,ensure_ascii=False)
# print(response.json())

end_time = time.time()
total_time = end_time - start_time

print(f"\n=== E2E 总体统计 ===")
print(f"总处理数量: {success_count}")
print(f"总耗时: {total_time:.2f}秒")
print(f"平均每条耗时: {total_time/success_count:.3f}秒")
print(f"每秒处理速度: {success_count/total_time:.2f} 条/秒")
