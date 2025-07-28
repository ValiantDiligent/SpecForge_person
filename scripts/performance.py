import requests
import json
import time
import threading
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, wait
import statistics
from transformers import AutoTokenizer

# ========== 参数解析 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='QPS测试工具')
    parser.add_argument('--port', type=int, default=30000, help='服务端口号 (默认: 30000)')
    parser.add_argument('--nums', type=int, default=100, help='处理的总数量 (默认: 100)')
    parser.add_argument('--qps', type=int, default=10, help='每秒发送请求数 (默认: 10)')
    parser.add_argument('--path', type=str,
                        default="/root/SecForce/SpecForge/cache/dataset/test725.jsonl",
                        help='数据文件路径')
    parser.add_argument('--timeout', type=int, default=300, help='请求超时时间(秒) (默认: 300)')
    return parser.parse_args()


# ========== 初始化参数 ==========
args = parse_args()
PORT = args.port
NUMS = args.nums
QPS = args.qps
path = args.path
TIMEOUT = args.timeout
url = f"http://localhost:{PORT}/v1/chat/completions"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 统计变量
success_count = 0
failed_count = 0
sent_count = 0
total_input_tokens = 0
total_output_tokens = 0
total_decoded_chunks = 0
request_queue = Queue()
tokenizer_futures = []

# 性能统计
e2e_times = []
ttft_times = []
tpot_times = []
response_lock = threading.Lock()

# ========== 限速器 ==========
class TokenBucketRateLimiter:
    def __init__(self, rate_per_second):
        self.capacity = rate_per_second
        self.tokens = rate_per_second
        self.fill_rate = rate_per_second
        self.timestamp = time.time()
        self.lock = threading.Lock()

    def consume(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
            self.timestamp = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

rate_limiter = TokenBucketRateLimiter(QPS)

# Tokenizer异步线程池
tokenizer_executor = ThreadPoolExecutor(max_workers=4)

def async_token_count(text, is_input):
    global total_input_tokens, total_output_tokens
    try:
        token_len = len(tokenizer.encode(text))
        with response_lock:
            if is_input:
                total_input_tokens += token_len
            else:
                total_output_tokens += token_len
    except Exception as e:
        print(f"Tokenizer 异常: {e}")

# ========== 请求函数 ==========
def send_request(data_dict):
    global success_count, failed_count, total_decoded_chunks
    request_start_time = time.time()
    ttft = None
    token_count = 0
    first_chunk_received = False

    try:
        # 异步统计输入 token
        input_text = json.dumps(data_dict["messages"])
        tokenizer_futures.append(tokenizer_executor.submit(async_token_count, input_text, True))

        response = requests.post(url, json=data_dict, timeout=TIMEOUT, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta and delta['content']:
                                    if not first_chunk_received:
                                        ttft = time.time() - request_start_time
                                        first_chunk_received = True

                                    tokenizer_futures.append(
                                        tokenizer_executor.submit(async_token_count, delta['content'], False)
                                    )
                                    total_decoded_chunks += 1
                        except json.JSONDecodeError:
                            continue

            e2e_time = time.time() - request_start_time
            tpot = None
            if token_count > 0 and ttft is not None:
                generation_time = e2e_time - ttft
                tpot = generation_time / token_count if generation_time > 0 else 0

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

# ========== QPS调度器 ==========
def qps_scheduler():
    global sent_count
    with ThreadPoolExecutor(max_workers=50) as executor:
        while True:
            if request_queue.empty():
                time.sleep(0.01)
                continue
            data_dict = request_queue.get()
            if data_dict is None:
                break
            while not rate_limiter.consume():
                time.sleep(0.001)
            executor.submit(send_request, data_dict)
            sent_count += 1

# ========== 主执行逻辑 ==========
print(f"开始处理，目标数量: {NUMS}，QPS: {QPS}")
start_time = time.time()
scheduler_thread = threading.Thread(target=qps_scheduler)
scheduler_thread.start()

num = 0
with open(path, "r") as f:
    for line in f:
        if num >= NUMS:
            break
        line = line.strip()
        if not line:
            continue
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

        request_queue.put(data_dict)
        num += 1

request_queue.put(None)
scheduler_thread.join()

# 等待所有请求完成
while (success_count + failed_count) < sent_count:
    time.sleep(0.1)

# 等待所有 token 统计任务完成
wait(tokenizer_futures)

end_time = time.time()
total_time = end_time - start_time

# ========== 统计输出 ==========
def safe_percentile(data, percentile):
    if not data:
        return 0
    return statistics.quantiles(data, n=100)[percentile-1] if len(data) > 1 else data[0]

def safe_avg(data):
    return statistics.mean(data) if data else 0

print(f"\n=== QPS 控制统计 ===")
print(f"目标QPS: {QPS}")
print(f"实际QPS: {sent_count/total_time:.2f}")
print(f"总发送数量: {sent_count}")
print(f"成功响应数: {success_count}")
print(f"失败响应数: {failed_count}")
print(f"响应成功率: {success_count/(success_count+failed_count)*100:.2f}%")
print(f"总耗时: {total_time:.2f}秒")

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

print(f"\n=== Token 统计 ===")
print(f"总输入 token 数: {total_input_tokens}")
print(f"总输出 token 数: {total_output_tokens}")
print(f"总 Token 数（输入 + 输出）: {total_input_tokens + total_output_tokens}")
print(f"总 decode 块数量（delta.content 数）: {total_decoded_chunks}")
