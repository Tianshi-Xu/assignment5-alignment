import os
import json
import tqdm  # 使用tqdm库来显示进度条，更友好

def convert_math_to_jsonl(base_dir, output_dir):
    """
    将 MATH 数据集的原始文件夹格式转换为 JSON Lines (.jsonl) 格式。

    参数:
    base_dir (str): 包含 'train' 和 'test' 文件夹的 MATH 数据集根目录。
    output_dir (str): 保存转换后 .jsonl 文件的目录。
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 需要处理的数据集划分 (splits)
    splits = ['train', 'test']
    
    for split in splits:
        print(f"--- Processing '{split}' split ---")
        
        split_path = os.path.join(base_dir, split)
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        if not os.path.isdir(split_path):
            print(f"Warning: Directory not found for split '{split}': {split_path}")
            continue
            
        all_records = []
        
        # 获取所有类别的子文件夹
        categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        # 使用 tqdm 显示进度
        for category in tqdm.tqdm(categories, desc=f"Scanning categories in '{split}'"):
            category_path = os.path.join(split_path, category)
            
            # 遍历类别文件夹下的所有 .json 文件
            for filename in os.listdir(category_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(category_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 创建一个新的记录，包含所有需要的信息
                        record = {
                            "problem": data.get("problem", ""),
                            "level": data.get("level", ""),
                            "type": data.get("type", category), # 优先使用文件内的type，否则用文件夹名
                            "solution": data.get("solution", "")
                        }
                        all_records.append(record)
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from file: {file_path}")
                    except Exception as e:
                        print(f"An error occurred while processing {file_path}: {e}")

        # 将所有记录写入 .jsonl 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(json.dumps(record) + '\n')
                
        print(f"Successfully created {output_file} with {len(all_records)} records.")

# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 请在这里修改您的路径 ---
    # 您手动下载的 MATH 数据集所在的根目录
    # 例如：'D:/downloads/math' 或者 '/home/user/datasets/math'
    # 这个目录下应该能看到 train 和 test 两个文件夹
    BASE_DATA_DIR = 'data/MATH' 

    # 您希望保存转换后文件的目录
    OUTPUT_DIR = 'data/math_hf'
    
    # 检查是否安装了tqdm
    try:
        import tqdm
    except ImportError:
        print("tqdm library not found. Progress bars will not be shown.")
        print("You can install it by running: pip install tqdm")
        
    # 运行转换函数
    convert_math_to_jsonl(BASE_DATA_DIR, OUTPUT_DIR)