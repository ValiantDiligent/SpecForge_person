import requests
import json
import time
import threading
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import statistics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='QPS测试工具')
    parser.add_argument('--port', type=int, default=30000, 
                       help='服务端口号 (默认: 30000)')
    parser.add_argument('--nums', type=int, default=100, 
                       help='处理的总数量 (默认: 100)')
    parser.add_argument('--qps', type=int, default=10, 
                       help='每秒发送请求数 (默认: 3)')
    parser.add_argument('--path', type=str, 
                       default="/root/SecForce/SpecForge/cache/dataset/test725.jsonl",
                       help='数据文件路径')
    parser.add_argument('--timeout', type=int, default=300,
                       help='请求超时时间(秒) (默认: 300)')
    return parser.parse_args()


args = parse_args()
PORT = args.port
NUMS = args.nums
QPS = args.qps
path = args.path
TIMEOUT = args.timeout
url = f"http://localhost:{PORT}/v1/chat/completions"

success_count = 0
failed_count = 0
sent_count = 0
request_queue = Queue()

# 性能统计数据
e2e_times = []  # 端到端时间
ttft_times = []  # 首字节时间
tpot_times = []  # 每令牌输出时间
response_lock = threading.Lock()  # 保护统计数据的锁

def send_request(data_dict):
    """发送请求的函数"""
    global success_count, failed_count, e2e_times, ttft_times, tpot_times
    
    request_start_time = time.time()
    ttft = None
    token_count = 0
    first_chunk_received = False
    
    try:
        response = requests.post(url, json=data_dict, timeout=TIMEOUT, stream=True)
        if response.status_code == 200:
            chunk_start_time = time.time()
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]  # 移除 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    # 记录首字节时间
                                    if not first_chunk_received:
                                        ttft = time.time() - request_start_time
                                        first_chunk_received = True
                                    
                                    # 统计token数量（简单估算）
                                    token_count += len(delta['content'].split())
                        except json.JSONDecodeError:
                            continue
            
            e2e_time = time.time() - request_start_time
            
            # 计算TPOT (Time Per Output Token)
            tpot = None
            if token_count > 0 and ttft is not None:
                generation_time = e2e_time - ttft
                tpot = generation_time / token_count if generation_time > 0 else 0
            
            # 线程安全地更新统计数据
            with response_lock:
                success_count += 1
                e2e_times.append(e2e_time)
                if ttft is not None:
                    ttft_times.append(ttft)
                if tpot is not None:
                    tpot_times.append(tpot)
        else:
            with response_lock:
                failed_count += 1
            print(f"请求失败，状态码: {response.status_code}")
    except Exception as e:
        with response_lock:
            failed_count += 1
        print(f"请求异常: {e}")

def qps_scheduler():
    """QPS调度器，控制发送频率"""
    global sent_count
    interval = 1.0 / QPS  # 每个请求之间的间隔时间
    
    with ThreadPoolExecutor(max_workers=50) as executor:  # 异步发送请求
        while True:
            if request_queue.empty():
                time.sleep(0.05)
                continue
                
            data_dict = request_queue.get()
            if data_dict is None:  # 结束信号
                break
                
            # 异步发送请求
            executor.submit(send_request, data_dict)
            sent_count += 1
            
            # 控制QPS
            time.sleep(interval)

# 读取数据并放入队列
print(f"开始处理，目标数量: {NUMS}，QPS: {QPS}")
start_time = time.time()

# 启动QPS调度器线程
scheduler_thread = threading.Thread(target=qps_scheduler)
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
            "temperature": 0,
            "stream": True
        }
        
        # 将请求放入队列
        request_queue.put(data_dict)

# 发送结束信号
request_queue.put(None)
scheduler_thread.join()
end_time = time.time()
total_time = end_time - start_time
print(f"实际QPS: {sent_count/total_time:.2f}")
print("等待所有请求处理完成...")
while (success_count + failed_count) < sent_count:
    time.sleep(0.1)
    current_finished = success_count + failed_count
    # print(f"进度: {current_finished}/{sent_count} ({current_finished/sent_count*100:.1f}%)")

response_time = time.time() - start_time

def safe_percentile(data, percentile):
    """安全计算百分位数"""
    if not data:
        return 0
    return statistics.quantiles(data, n=100)[percentile-1] if len(data) > 1 else data[0]

def safe_avg(data):
    """安全计算平均值"""
    return statistics.mean(data) if data else 0

print(f"\n=== QPS 控制统计 ===")
print(f"目标QPS: {QPS}")
print(f"实际QPS: {sent_count/total_time:.2f}")
print(f"总发送数量: {sent_count}")
print(f"成功响应数: {success_count}")
print(f"失败响应数: {failed_count}")
print(f"响应成功率: {success_count/(success_count+failed_count)*100:.2f}%")
print(f"总耗时: {response_time:.2f}秒")

# 性能指标统计
print(f"\n=== 性能指标统计 ===")
if e2e_times:
    print(f"E2E (端到端时间):")
    print(f"  平均值: {safe_avg(e2e_times):.3f}s")
    print(f"  P50: {safe_percentile(e2e_times, 50):.3f}s")
    print(f"  P90: {safe_percentile(e2e_times, 90):.3f}s")
    print(f"  P95: {safe_percentile(e2e_times, 95):.3f}s")
    print(f"  P99: {safe_percentile(e2e_times, 99):.3f}s")

if ttft_times:
    print(f"TTFT (首字节时间):")
    print(f"  平均值: {safe_avg(ttft_times):.3f}s")
    print(f"  P50: {safe_percentile(ttft_times, 50):.3f}s")
    print(f"  P90: {safe_percentile(ttft_times, 90):.3f}s")
    print(f"  P95: {safe_percentile(ttft_times, 95):.3f}s")
    print(f"  P99: {safe_percentile(ttft_times, 99):.3f}s")

if tpot_times:
    print(f"TPOT (每令牌输出时间):")
    print(f"  平均值: {safe_avg(tpot_times):.3f}s")
    print(f"  P50: {safe_percentile(tpot_times, 50):.3f}s")
    print(f"  P90: {safe_percentile(tpot_times, 90):.3f}s")
    print(f"  P95: {safe_percentile(tpot_times, 95):.3f}s")
    print(f"  P99: {safe_percentile(tpot_times, 99):.3f}s")
# print(f"待处理请求: {request_queue.qsize()}")