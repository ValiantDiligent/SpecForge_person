import os
import subprocess
import itertools
import time
from datetime import datetime

# 测试参数配置
ports = {
    30000: "qwen3",
    30001: "qwen3-eagle3-open",
    30002: "qwen3-eagle3-ours"
}
qps_list = [15, 10, 5]
nums_list = [1000, 3000, 5000]

# 日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 命令模板
CMD_TEMPLATE = "python3 performance.py --nums {nums} --qps {qps} --port {port}"

# 任务生成（每个port一个队列）
tasks_by_port = {port: [] for port in ports}
for port in ports:
    for qps, nums in itertools.product(qps_list, nums_list):
        tasks_by_port[port].append((qps, nums))

# 执行任务（串行 per port）
def run_task(port, qps, nums):
    tag = ports[port]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{tag}_port{port}_qps{qps}_nums{nums}_{timestamp}.log")

    cmd = CMD_TEMPLATE.format(nums=nums, qps=qps, port=port)
    print(f"开始任务: {cmd}")
    print(f"日志: {log_file}")

    with open(log_file, "w") as f:
        f.write(f"=== CMD: {cmd} ===\n")
        f.write(f"=== START: {datetime.now()} ===\n\n")
        f.flush()

        # subprocess 实时输出
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(f"[PORT {port}] {line.strip()}")
            f.write(line)
            f.flush()

        process.wait()
        f.write(f"\n=== END: {datetime.now()} ===\n")
        f.flush()

        if process.returncode != 0:
            print(f"⚠️ 错误: 命令执行失败，退出码: {process.returncode}")
        else:
            print(f"✅ 完成: {cmd}")

# 主调度逻辑
if __name__ == "__main__":
    for port, task_list in tasks_by_port.items():
        print(f"\n🌀 开始串行任务队列 for PORT {port} ({ports[port]}), 共 {len(task_list)} 项\n")
        for qps, nums in task_list:
            run_task(port, qps, nums)
            print(f"✅ 完成一个任务 [{port}] qps={qps} nums={nums}")
            print("-" * 60)
            time.sleep(3)  # 稳定性间隔，可去掉
