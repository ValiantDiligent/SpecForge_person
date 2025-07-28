import os
import subprocess
import time
from datetime import datetime
import threading

# === 配置项 ===
PORT_CONFIG = {
    30000: "qwen3",
    30001: "qwen3_eagle_open",
    30002: "qwen3_eagle_ours"
}

QPS_LIST = [15, 10, 5]
NUMS_LIST = [1000, 3000, 5000]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# === 执行单个任务，并记录日志 ===
def run_task(port, name, qps, nums):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_port{port}_qps{qps}_nums{nums}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    cmd = f"python3 performance.py --nums {nums} --qps {qps} --port {port}"
    print(f"\n__ Running for PORT {port} ({name}) : QPS={qps}, NUMS={nums} __")
    print(f"Command: {cmd}")
    print(f"Log file: {log_path}")

    with open(log_path, "w") as logfile:
        logfile.write(f"=== Running: {cmd} ===\n")
        logfile.flush()

        # subprocess with live output to log
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        for line in process.stdout:
            logfile.write(line)
            logfile.flush()  # 💡 保证中途 crash 也有日志
        process.wait()

        logfile.write(f"\n=== Task finished: {cmd} ===\n")
        logfile.flush()

# === 每个 port 独立线程串行调度 ===
def run_all_for_port(port, name):
    for qps in QPS_LIST:
        for nums in NUMS_LIST:
            run_task(port, name, qps, nums)

# === 主程序 ===
def main():
    threads = []
    for port, name in PORT_CONFIG.items():
        t = threading.Thread(target=run_all_for_port, args=(port, name))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    for t in threads:
        t.join()

    print("\n✅ 所有任务执行完成")

if __name__ == "__main__":
    main()
