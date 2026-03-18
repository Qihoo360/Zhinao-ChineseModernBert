#!/usr/bin/env python3
# eval_clue.py

import os
import csv
import subprocess
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import GPUtil
from queue import Queue
import threading

# CLUE任务列表（与clue_evaluator.py中一致）
CLUE_TASKS = [
    "afqmc", "cmnli", "csl", "iflytek", "ocnli", "tnews", "wsc", "c3"
]

def get_available_gpus():
    """
    获取当前可用的GPU数量
    """
    try:
        # 使用GPUtil获取空闲GPU
        gpus = GPUtil.getGPUs()
        available_gpus = []
        for gpu in gpus:
            # 如果GPU使用率低于10%且显存使用低于10%，认为是空闲的
            if gpu.load < 0.1 and gpu.memoryUtil < 0.1:
                available_gpus.append(gpu.id)
        return available_gpus
    except Exception:
        # 如果无法获取GPU信息，返回CPU模式
        return []

def run_task_on_gpu(model_path, task_name, gpu_id):
    """
    在指定GPU上运行单个CLUE任务评估
    """
    cmd = [
        "python", "clue_evaluator.py",
        "--model_name_or_path", model_path,
        "--task_name", task_name,
        "--output_dir", f"./clue_results_batch"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)  # 2小时超时
        if result.returncode == 0:
            return task_name, "success", result.stdout
        else:
            return task_name, "failed", result.stderr
    except subprocess.TimeoutExpired:
        return task_name, "timeout", "Process timed out"
    except Exception as e:
        return task_name, "error", str(e)

def collect_results(model_path):
    """
    收集所有任务的评估结果并汇总成CSV
    """
    all_results = []
    
    for task in CLUE_TASKS:
        result_file = os.path.join("./clue_results_batch", f"{task}_results.txt")
        
        if result_file and os.path.exists(result_file):
            with open(result_file, 'r') as f:
                lines = f.readlines()
                
            result_dict = {"task": task}
            for line in lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    result_dict[key.strip()] = value.strip()
            
            all_results.append(result_dict)
        else:
            all_results.append({"task": task, "status": "missing"})
    
    # 保存为CSV文件
    if os.path.exists(model_path):
        csv_file = os.path.join(model_path, "clue_evaluation_results.csv")
    else:
        dirname = os.path.join("./clue_results_online_model", model_path)
        os.makedirs(dirname, exist_ok=True)
        csv_file = os.path.join(dirname, "clue_evaluation_results.csv")
    if all_results:
        fieldnames = set()
        for result in all_results:
            fieldnames.update(result.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    
    return csv_file, all_results

class GPUManager:
    def __init__(self, available_gpus):
        self.available_gpus = available_gpus if available_gpus else [None]
        self.gpu_queue = Queue()
        for gpu in self.available_gpus:
            self.gpu_queue.put(gpu)
        self.lock = threading.Lock()
    
    def acquire_gpu(self):
        """获取一个可用的GPU"""
        try:
            return self.gpu_queue.get(timeout=1)
        except:
            return None
    
    def release_gpu(self, gpu_id):
        """释放GPU资源"""
        self.gpu_queue.put(gpu_id)

def worker(gpu_manager, model_path, task_queue, results):
    """
    工作线程函数：持续从任务队列中获取任务并在可用GPU上执行
    """
    while True:
        try:
            # 尝试获取任务
            task_name = task_queue.get_nowait()
        except:
            # 任务队列已空
            break
            
        # 获取可用GPU
        gpu_id = gpu_manager.acquire_gpu()
        if gpu_id is None:
            # 如果暂时没有GPU可用，放回任务并稍后再试
            task_queue.put(task_name)
            time.sleep(5)
            continue
            
        print(f"开始执行任务 {task_name} on GPU {gpu_id}")
        
        # 执行任务
        result = run_task_on_gpu(model_path, task_name, gpu_id)
        results.append(result)
        
        # 释放GPU
        gpu_manager.release_gpu(gpu_id)
        
        # 标记任务完成
        task_queue.task_done()
        
        print(f"任务 {task_name} 在 GPU {gpu_id} 上执行完成，状态: {result[1]}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on all CLUE tasks in parallel")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Max number of parallel workers (default: auto-detect available GPUs)")
    args = parser.parse_args()
    
    # 获取可用GPU
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        print("No available GPUs detected, falling back to CPU mode (will be slow)")
        available_gpus = [None]  # CPU模式
    
    max_workers = args.max_workers or len(available_gpus)
    print(f"Detected available GPUs: {available_gpus}")
    print(f"Using up to {max_workers} worker threads")
    
    # 清理并创建基础输出目录
    if os.path.exists("./clue_results_batch"):
        import shutil
        shutil.rmtree("./clue_results_batch")
    os.makedirs("./clue_results_batch", exist_ok=True)
    
    # 创建任务队列
    task_queue = Queue()
    for task in CLUE_TASKS:
        task_queue.put(task)
    
    # 初始化GPU管理器
    gpu_manager = GPUManager(available_gpus)
    
    # 存储结果
    results = []
    
    # 启动工作线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _ in range(max_workers):
            future = executor.submit(worker, gpu_manager, args.model_path, task_queue, results)
            futures.append(future)
        
        # 等待所有任务完成
        for future in as_completed(futures):
            pass  # 工作线程完成后会自动退出
    
    # 输出结果
    print("\n所有任务执行完成，结果如下:")
    for task_name, status, output in results:
        print(f"Task {task_name} finished with status: {status}")
        if status != "success":
            print(f"Output: {output[:200]}...")  # 只显示前200个字符
    
    # 收集并汇总结果
    csv_file, summary_results = collect_results(args.model_path)
    print(f"\nEvaluation complete! Results saved to: {csv_file}")
    
    # 打印摘要
    print("\nSummary:")
    sum = 0.0
    for result in summary_results:
        task = result.get("task", "unknown")
        accuracy = result.get("eval_accuracy", "N/A")
        print(f"  {task}: {accuracy}")
        sum += accuracy
    print(f"  Average: {sum / len(summary_results)}")

if __name__ == "__main__":
    main()