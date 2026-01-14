import os
import shutil
import wespeaker
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 配置区域 =================
# 强制 CPU (必须在最上面设置)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WESPEAKER_DEVICE"] = "cpu"

PROXY_DIR = "clips_16k"     # AI 读这里 (16k)
HQ_DIR = "clips_HQ"         # 搬运工搬这里 (44.1k)
RESULT_DIR = "final_result" # 结果存这里
EXAMPLES_DIR = "examples"   # 注册音频目录
MODEL_PATH = 'models'       # 模型名称 (必须是 english/resnet)
SCORE_THRESHOLD = 0.7    # 阈值

# 进程数：默认使用 CPU 核心数减 2 (留点资源给系统)
# 如果你想要火力全开，改成 os.cpu_count()
WORKER_NUM = max(1, os.cpu_count() - 2) 
# ===========================================

# 全局变量（用于在子进程中共享模型和声纹库）
worker_model = None
worker_speakers = None

def compute_cosine_similarity(embed1, embed2):
    e1 = embed1.detach().numpy().flatten()
    e2 = embed2.detach().numpy().flatten()
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(e1, e2) / (norm1 * norm2)

def init_worker(model_path, speakers_emb_dict):
    """
    子进程初始化函数：
    每个进程启动时，都会运行一次这个函数。
    用来加载模型和声纹库，避免重复传递数据。
    """
    global worker_model, worker_speakers
    
    # 每个进程都屏蔽显卡，强制 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["WESPEAKER_DEVICE"] = "cpu"
    
    # 加载模型
    # print(f"进程 {os.getpid()} 正在加载模型...")
    worker_model = wespeaker.load_model(model_path)
    worker_model.set_device('cpu')
    
    # 接收主进程传来的主角声纹
    worker_speakers = speakers_emb_dict

def process_single_file(proxy_path):
    """
    单个文件的处理逻辑 (将在子进程中运行)
    """
    global worker_model, worker_speakers
    
    try:
        # 1. 提取特征
        clip_emb = worker_model.extract_embedding(proxy_path)
        
        # 2. 静音检测
        if np.linalg.norm(clip_emb.detach().numpy()) < 0.1:
            return False # 跳过

        # 3. 算分比对
        best_score = -1.0
        best_name = "Unknown"
        
        for spk_name, spk_emb in worker_speakers.items():
            score = compute_cosine_similarity(clip_emb, spk_emb)
            if score > best_score:
                best_score = score
                best_name = spk_name
        
        # 4. 阈值判定
        target_folder = "Unknown"
        if best_score >= SCORE_THRESHOLD:
            target_folder = best_name
        
        # 5. 寻找并复制 HQ 文件
        # 计算相对路径
        # 假设 PROXY_DIR="clips_16k", proxy_path="clips_16k/01/001.wav"
        # rel_path = "01/001.wav"
        # 这种切分方式比 os.path.relpath 更稳健一点，防止多进程下路径错乱
        if PROXY_DIR in proxy_path:
             rel_path = proxy_path.split(PROXY_DIR)[1].lstrip(os.sep)
        else:
             rel_path = os.path.basename(proxy_path) # Fallback
             
        hq_source_path = os.path.join(HQ_DIR, rel_path)
        
        if os.path.exists(hq_source_path):
            parent_folder = os.path.basename(os.path.dirname(proxy_path))
            filename = os.path.basename(proxy_path)
            new_name = f"{parent_folder}_{filename}"
            
            dst_path = os.path.join(RESULT_DIR, target_folder, new_name)
            
            # 执行复制
            shutil.copy(hq_source_path, dst_path)
            return True
        else:
            return False # 没找到 HQ 文件

    except Exception:
        return False

def main():
    print(f"🔥 启动多进程加速 (使用 {WORKER_NUM} 个 CPU 核心)...")
    
    # 1. 在主进程加载一次模型，为了注册主角
    print("正在主进程注册主角声纹...")
    temp_model = wespeaker.load_model(MODEL_PATH)
    temp_model.set_device('cpu')
    
    speakers_emb = {}
    if not os.path.exists(EXAMPLES_DIR):
        print(f"❌ 错误: 找不到 {EXAMPLES_DIR}")
        return

    for file in os.listdir(EXAMPLES_DIR):
        if file.lower().endswith('.wav'):
            name = os.path.splitext(file)[0]
            path = os.path.join(EXAMPLES_DIR, file)
            emb = temp_model.extract_embedding(path)
            if np.linalg.norm(emb.detach().numpy()) > 0.1:
                speakers_emb[name] = emb
                print(f"  - 已注册: {name}")
    
    del temp_model # 释放主进程模型，节省内存
    
    if not speakers_emb:
        print("❌ 没有注册样本！")
        return

    # 准备目录
    os.makedirs(os.path.join(RESULT_DIR, "Unknown"), exist_ok=True)
    for name in speakers_emb.keys():
        os.makedirs(os.path.join(RESULT_DIR, name), exist_ok=True)

    # 2. 扫描文件任务
    print("正在扫描文件列表...")
    task_files = []
    for root, dirs, files in os.walk(PROXY_DIR):
        for file in files:
            if file.lower().endswith('.wav'):
                task_files.append(os.path.join(root, file))
    
    print(f"共找到 {len(task_files)} 个任务，开始并行处理...")

    # 3. 启动进程池
    # initargs 负责把声纹库传给每个子进程
    with ProcessPoolExecutor(max_workers=WORKER_NUM, 
                             initializer=init_worker, 
                             initargs=(MODEL_PATH, speakers_emb)) as executor:
        
        # 提交所有任务
        futures = [executor.submit(process_single_file, f) for f in task_files]
        
        success_count = 0
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                success_count += 1

    print(f"\n🎉 全部完成！已处理并归类 {success_count} 个文件到 {RESULT_DIR}")

if __name__ == "__main__":
    # Windows/WSL 下必须加这行
    multiprocessing.set_start_method('spawn', force=True)
    main()
