from openai import OpenAI
from typing import List, Dict, Optional
import json
import time
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import signal
import sys


# 多个deployment配置
deployment_names = [
    "DeepSeek-R1-0528",
    "DeepSeek-R1-0528-2",
    "DeepSeek-R1-0528-3",
    "DeepSeek-R1-0528-4",
    "DeepSeek-R1-0528-5",
    "DeepSeek-R1-0528-6",
    "DeepSeek-R1-0528-7",
    "DeepSeek-R1-0528-8",
    "DeepSeek-R1-0528-9",
]

# 创建多个客户端（每个deployment一个）
clients = []
for i, deployment in enumerate(deployment_names):
    client = OpenAI(
        base_url=f"{endpoint}",
        api_key=api_key
    )
    clients.append({
        'client': client,
        'deployment_id': i,
        'deployment_name': deployment,
        'success_count': 0,
        'failure_count': 0,
        'last_used': 0,
        'is_healthy': True
    })

# 用于轮询的索引和锁
import random
current_client_index = random.randint(0, len(clients) - 1)
client_lock = threading.Lock()

# 错误统计
error_stats = {
    'TIMEOUT': 0,
    'RATE_LIMIT': 0,
    'AUTH_ERROR': 0,
    'NOT_FOUND': 0,
    'SERVICE_UNAVAILABLE': 0,
    'BAD_REQUEST': 0,
    'CONNECTION_ERROR': 0,
    'SSL_ERROR': 0,
    'UNKNOWN': 0
}

def get_next_client():
    """获取下一个可用的客户端，使用轮询策略"""
    global current_client_index
    
    with client_lock:
        # 首先尝试找到健康的客户端
        healthy_clients = [i for i, c in enumerate(clients) if c['is_healthy']]
        
        if not healthy_clients:
            print("Warning: No healthy clients available, using all clients")
            healthy_clients = list(range(len(clients)))
        
        # 轮询到下一个健康的客户端
        start_index = current_client_index
        attempts = 0
        
        while attempts < len(healthy_clients):
            current_client_index = (current_client_index + 1) % len(clients)
            if current_client_index in healthy_clients:
                break
            attempts += 1
        
        client_info = clients[current_client_index]
        client_info['last_used'] = time.time()
        
        return client_info

def mark_client_status(client_info: Dict, success: bool, error_msg: str = ""):
    """标记客户端状态"""
    with client_lock:
        if success:
            client_info['success_count'] += 1
            client_info['is_healthy'] = True
        else:
            client_info['failure_count'] += 1
            
            # 如果连续失败太多，标记为不健康
            total_requests = client_info['success_count'] + client_info['failure_count']
            if total_requests > 10 and client_info['failure_count'] / total_requests > 0.8:
                client_info['is_healthy'] = False
                print(f"Marking deployment {client_info['deployment_id']} ({client_info['deployment_name']}) as unhealthy due to high failure rate")

def print_client_stats():
    """打印客户端统计信息"""
    print("\n=== Deployment Statistics ===")
    for client_info in clients:
        total = client_info['success_count'] + client_info['failure_count']
        success_rate = client_info['success_count'] / total * 100 if total > 0 else 0
        status = "✓ Healthy" if client_info['is_healthy'] else "✗ Unhealthy"
        print(f"Deployment {client_info['deployment_id']} ({client_info['deployment_name']}): {client_info['success_count']}/{total} success ({success_rate:.1f}%) - {status}")
    
    print("\n=== Error Statistics ===")
    total_errors = sum(error_stats.values())
    if total_errors > 0:
        for error_type, count in error_stats.items():
            if count > 0:
                percentage = count / total_errors * 100
                print(f"{error_type}: {count} ({percentage:.1f}%)")
    else:
        print("No errors recorded")

def load_math_dataset(jsonl_file: str) -> List[Dict]:
    """加载 MATH 数据集"""
    examples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def format_prompt(prompt_file: str, problem: str) -> str:
    """使用模板格式化问题"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read().strip()

    return template.format(question=problem)

def get_completion(prompt: str, max_retries: int = 3, delay: float = 1.0, timeout: int = 60) -> Optional[str]:
    """获取单个prompt的API响应，带重试机制和多API key支持"""
    last_error = None
    
    for attempt in range(max_retries):
        # 获取下一个可用的客户端
        client_info = get_next_client()
        client = client_info['client']
        
        try:
            completion = client.chat.completions.create(
                model=client_info['deployment_name'],  # 使用对应的deployment名称
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.7,
                max_tokens=4096,
                timeout=timeout
            )
            
            # 标记成功
            mark_client_status(client_info, success=True)
            return completion.choices[0].message.content
            
        except Exception as e:
            last_error = e
            
            # 分类错误类型
            error_type = type(e).__name__
            error_str = str(e)
            
            if "timeout" in error_str.lower():
                error_category = "TIMEOUT"
            elif "rate limit" in error_str.lower() or "429" in error_str or "too many requests" in error_str.lower():
                error_category = "RATE_LIMIT"
            elif "401" in error_str or "unauthorized" in error_str.lower() or "invalid api key" in error_str.lower():
                error_category = "AUTH_ERROR"
            elif "404" in error_str or "not found" in error_str.lower() or "deployment not found" in error_str.lower():
                error_category = "NOT_FOUND"
            elif "503" in error_str or "service unavailable" in error_str.lower() or "server error" in error_str.lower():
                error_category = "SERVICE_UNAVAILABLE"
            elif "400" in error_str or "bad request" in error_str.lower() or "invalid request" in error_str.lower():
                error_category = "BAD_REQUEST"
            elif "connection" in error_str.lower() or "network" in error_str.lower() or "dns" in error_str.lower():
                error_category = "CONNECTION_ERROR"
            elif "ssl" in error_str.lower() or "certificate" in error_str.lower() or "tls" in error_str.lower():
                error_category = "SSL_ERROR"
            elif "quota" in error_str.lower() or "limit" in error_str.lower():
                error_category = "RATE_LIMIT"
            elif "model" in error_str.lower() and ("not available" in error_str.lower() or "not supported" in error_str.lower()):
                error_category = "NOT_FOUND"
            else:
                error_category = "UNKNOWN"
            
            error_msg = f"Attempt {attempt + 1} with deployment {client_info['deployment_id']} ({client_info['deployment_name']}) failed: [{error_category}] {error_type}: {error_str[:100]}..."
            print(error_msg)
            
            # 更新错误统计
            error_stats[error_category] += 1
            
            # 标记失败
            mark_client_status(client_info, success=False, error_msg=f"{error_category}: {error_str[:100]}")
            
            if attempt < max_retries - 1:
                # 短暂等待，但不要太长，因为我们会换key
                time.sleep(delay * (1.5 ** attempt))
            else:
                print(f"Failed to get response after {max_retries} attempts with different deployments")
                print(f"Last error: {last_error}")
    
    return None

def get_completion_with_index(args) -> Dict:
    """带索引的completion函数，用于并行处理"""
    index, prompt = args
    try:
        response = get_completion(prompt, timeout=120)  # API响应超时2分钟
        return {
            "prompt_index": index,
            "prompt": prompt,
            "response": response,
            "success": response is not None,
            "error": None
        }
    except Exception as e:
        return {
            "prompt_index": index,
            "prompt": prompt,
            "response": None,
            "success": False,
            "error": str(e)[:200]
        }

def process_batch_parallel_robust(prompts: List[str], max_workers: int = 10, task_timeout: int = 300) -> List[Dict]:
    """更健壮的并行处理，带信号处理和更好的超时控制"""
    print(f"Processing {len(prompts)} prompts with {max_workers} parallel workers (robust version)")
    
    # 准备带索引的参数
    indexed_prompts = [(i, prompt) for i, prompt in enumerate(prompts)]
    results = [None] * len(prompts)  # 预分配结果数组
    completed_count = 0
    
    # 信号处理，允许优雅退出
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal. Shutting down gracefully...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(get_completion_with_index, args): args[0] 
            for args in indexed_prompts
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            try:
                for future in as_completed(future_to_index, timeout=task_timeout):
                    try:
                        result = future.result(timeout=120)  # 给每个结果10秒的获取时间
                        index = result['prompt_index']
                        results[index] = result
                        completed_count += 1
                        
                        if result['success']:
                            status = "✓"
                        else:
                            status = "✗"
                            if result.get('error'):
                                print(f"Task {index} failed: {result['error'][:50]}...")
                        
                        pbar.set_postfix({
                            'completed': completed_count,
                            'success_rate': f"{sum(1 for r in results if r and r['success'])}/{completed_count}"
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        # 处理超时或其他异常
                        index = future_to_index.get(future, -1)
                        if index >= 0:
                            error_result = {
                                "prompt_index": index,
                                "prompt": indexed_prompts[index][1] if index < len(indexed_prompts) else "",
                                "response": None,
                                "success": False,
                                "error": f"Future timeout or error: {str(e)[:100]}"
                            }
                            results[index] = error_result
                        print(f"Task {index} failed with timeout/error: {str(e)[:50]}...")
                        pbar.update(1)
                        
            except Exception as e:
                print(f"Overall timeout or error in batch processing: {e}")
                # 取消所有未完成的任务
                for future in future_to_index:
                    if not future.done():
                        future.cancel()
        
        # 确保所有未处理的索引都有结果
        for i, result in enumerate(results):
            if result is None:
                results[i] = {
                    "prompt_index": i,
                    "prompt": indexed_prompts[i][1] if i < len(indexed_prompts) else "",
                    "response": None,
                    "success": False,
                    "error": "Task was not completed (timeout or cancellation)"
                }
    
    # 过滤None值并确保所有结果都存在
    final_results = [r for r in results if r is not None]
    
    successful_count = sum(1 for r in final_results if r['success'])
    failed_count = len(final_results) - successful_count
    print(f"Processing completed: {successful_count}/{len(final_results)} successful, {failed_count} failed")
    
    if failed_count > 0:
        print("Failed tasks will be marked with error information in the output file")
    
    return final_results

def process_batch_parallel(prompts: List[str], max_workers: int = 10, task_timeout: int = 90) -> List[Dict]:
    """并行处理prompts并收集响应，带超时控制"""
    print(f"Processing {len(prompts)} prompts with {max_workers} parallel workers")
    
    # 准备带索引的参数
    indexed_prompts = [(i, prompt) for i, prompt in enumerate(prompts)]
    
    results = []
    completed_count = 0
    failed_futures = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(get_completion_with_index, args): args[0] 
            for args in indexed_prompts
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            for future in as_completed(future_to_index, timeout=task_timeout):
                try:
                    result = future.result(timeout=5)  # 给每个结果5秒的获取时间
                    results.append(result)
                    completed_count += 1
                    
                    if result['success']:
                        status = "✓"
                    else:
                        status = "✗"
                        if result.get('error'):
                            print(f"Task {result['prompt_index']} failed: {result['error'][:50]}...")
                    
                    pbar.set_postfix({
                        'completed': completed_count,
                        'success_rate': f"{sum(1 for r in results if r['success'])}/{len(results)}"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    # 处理超时或其他异常
                    index = future_to_index.get(future, -1)
                    error_result = {
                        "prompt_index": index,
                        "prompt": indexed_prompts[index][1] if index >= 0 and index < len(indexed_prompts) else "",
                        "response": None,
                        "success": False,
                        "error": f"Future timeout or error: {str(e)[:100]}"
                    }
                    results.append(error_result)
                    failed_futures.append(future)
                    print(f"Task {index} failed with timeout/error: {str(e)[:50]}...")
                    pbar.update(1)
        
        # 取消所有未完成的任务
        for future in future_to_index:
            if not future.done():
                future.cancel()
                print(f"Cancelled unfinished task")
    
    # 按索引排序结果
    results.sort(key=lambda x: x['prompt_index'])
    
    successful_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - successful_count
    print(f"Processing completed: {successful_count}/{len(results)} successful, {failed_count} failed")
    
    if failed_count > 0:
        print("Failed tasks will be marked with error information in the output file")
    
    return results

def process_batch(prompts: List[str], batch_size: int = 10, delay_between_requests: float = 0.5) -> List[Dict]:
    """批量处理prompts并收集响应（串行版本，保留作为备选）"""
    results = []
    
    print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch = prompts[i:i + batch_size]
        batch_results = []
        
        for j, prompt in enumerate(batch):
            print(f"Processing prompt {i + j + 1}/{len(prompts)}")
            response = get_completion(prompt)
            
            result = {
                "prompt_index": i + j,
                "prompt": prompt,
                "response": response,
                "success": response is not None
            }
            batch_results.append(result)
            results.append(result)
            
            # 避免过于频繁的API调用
            if j < len(batch) - 1:  # 不在最后一个请求后等待
                time.sleep(delay_between_requests)
        
        print(f"Batch {i//batch_size + 1} completed: {sum(1 for r in batch_results if r['success'])}/{len(batch_results)} successful")
    
    return results

def save_results(results: List[Dict], output_file: str):
    """保存结果到文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    successful_count = sum(1 for r in results if r['success'])
    print(f"Results saved to {output_file}")
    print(f"Total: {len(results)}, Successful: {successful_count}, Failed: {len(results) - successful_count}")

def test_single_prompt():
    """测试单个prompt的API调用"""
    print("=== Testing single prompt ===")
    
    # 使用一个简单的测试问题
    test_problem = "Determine the average of the integers $71,$ $72,$ $73,$ $74,$ $75."
    prompt_file = "cs336_alignment/prompts/plain.prompt"
    
    try:
        # 格式化prompt
        formatted_prompt = format_prompt(prompt_file, test_problem)
        print(f"Formatted prompt:\n{formatted_prompt}")
        
        # 测试API调用
        print("\nTesting API call...")
        start_time = time.time()
        response = get_completion(formatted_prompt, max_retries=2, timeout=120)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response:
            print(f"✓ Success! Response length: {len(response)} characters")
            print(f"Response preview:\n{response}")
            
            # 显示API key统计
            print_client_stats()
            return True
        else:
            print("✗ Failed to get response")
            print_client_stats()
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        print_client_stats()
        return False

def main():
    """主函数：批量处理MATH数据集"""
    prompt_file = "cs336_alignment/prompts/plain.prompt"
    jsonl_file = "data/math_hf/train.jsonl"
    output_file = "data/math_hf/train_math_responses.jsonl"
    
    # 加载数据集
    print("Loading MATH dataset...")
    examples = load_math_dataset(jsonl_file)
    print(f"Loaded {len(examples)} examples")
    
    # 格式化prompts
    print("Formatting prompts...")
    prompts = [format_prompt(prompt_file, example['problem']) for example in examples]
    
    # 并行批量处理
    print("Starting parallel batch processing...")
    results = process_batch_parallel_robust(
        prompts, 
        max_workers=32,  # 适中的并发数
        task_timeout=10800  # 10分钟总超时时间
    )
    
    # 保存结果
    save_results(results, output_file)
    
    # 打印API key统计信息
    print_client_stats()
    
    print("Processing completed!")

if __name__ == "__main__":
    # test_single_prompt()
    main()
  