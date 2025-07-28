import requests
import json
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

url = f"http://localhost:30000/v1/chat/completions"
path = "/root/SpecForge/cache/dataset/test725.jsonl"
NUMS = 1000
BF = 10  # 设置每次并发发送的请求

success_count = 0
failed_count = 0
sent_count = 0
request_queue = Queue()

# 添加锁保护全局变量
count_lock = threading.Lock()

def send_request(data_dict):
    """发送请求的函数"""
    global success_count, failed_count
    try:
        response = requests.post(url, json=data_dict, timeout=300)
        with count_lock:
            if response.status_code == 200:
                success_count += 1
            else:
                failed_count += 1
                print(f"请求失败，状态码: {response.status_code}")
    except Exception as e:
        with count_lock:
            failed_count += 1
            print(f"请求异常: {e}")

def BF_scheduler():
    """BF调度器，控制发送频率"""
    global sent_count
    
    with ThreadPoolExecutor(max_workers=100) as executor:  # 异步发送请求
        while True:
            batch_start_time = time.time()
            requests_sent_this_batch = 0
            
            # 每秒发送BF个请求
            for _ in range(BF):
                if request_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                data_dict = request_queue.get()
                if data_dict is None:  # 结束信号
                    return
                    
                # 异步发送请求
                executor.submit(send_request, data_dict)
                with count_lock:
                    sent_count += 1
                requests_sent_this_batch += 1
            
            # 如果没有发送任何请求，说明队列空了或遇到结束信号
            if requests_sent_this_batch == 0:
                time.sleep(0.05)
                continue
            
            # 控制并发
            while (success_count + failed_count) < sent_count:
                time.sleep(0.1)
            current_finished = success_count + failed_count
            
# 读取数据并放入队列
print(f"开始处理，目标数量: {NUMS}，BF: {BF}")
start_time = time.time()

# 启动BF调度器线程
scheduler_thread = threading.Thread(target=BF_scheduler)
scheduler_thread.start()

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
        if len(conversations_list) >= 3:
            conversations_list.pop()
        
        data_dict = {
            "model": "Qwen/Qwen3-8B",
            "messages": conversations_list,
            "temperature": 0
        }
        
        # 将请求放入队列
        request_queue.put(data_dict)

# 发送结束信号
request_queue.put(None)
scheduler_thread.join()
end_time = time.time()
total_time = end_time - start_time

print("等待所有请求处理完成...")
while (success_count + failed_count) < sent_count:
    time.sleep(0.1)
    current_finished = success_count + failed_count
    # print(f"进度: {current_finished}/{sent_count} ({current_finished/sent_count*100:.1f}%)")

response_time = time.time() - start_time

print(f"\n=== BF 控制统计 ===")
print(f"目标BF: {BF}")
print(f"实际QPS: {sent_count/total_time:.2f}")
print(f"总发送数量: {sent_count}")
print(f"成功响应数: {success_count}")
print(f"失败响应数: {failed_count}")
print(f"响应成功率: {success_count/(success_count+failed_count)*100:.2f}%")
print(f"总耗时: {response_time:.2f}秒")
# print(f"待处理请求: {request_queue.qsize()}")