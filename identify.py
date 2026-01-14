import os
import shutil
import sys
import types
import logging
from unittest.mock import MagicMock

# ================= 日志配置 =================
# 抑制 wespeaker 加载模型时的 "missing tensor" 警告
class WarningFilter(logging.Filter):
    def filter(self, record):
        return "missing tensor" not in record.getMessage()

logging.getLogger().addFilter(WarningFilter())

# ================= Windows 兼容性补丁 =================
# 在导入 wespeaker 之前，检查并 Mock torchaudio.sox_effects
# 这是一个针对 Windows 环境的临时修复，因为 Windows 版 torchaudio 不支持 sox_effects
try:
    import torchaudio
    # 修复 s3prl 调用 torchaudio.set_audio_backend 报错
    if not hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend = MagicMock()

    if not hasattr(torchaudio, 'sox_effects'):
        # 创建一个假的模块
        mock_sox = types.ModuleType('torchaudio.sox_effects')
        # Mock 常用函数，防止调用报错 (返回空值或不做任何操作)
        mock_sox.apply_effects_tensor = MagicMock(return_value=(None, None))
        mock_sox.apply_effects_file = MagicMock(return_value=(None, None))
        
        # 将其注入到 sys.modules 和 torchaudio 中
        sys.modules['torchaudio.sox_effects'] = mock_sox
        torchaudio.sox_effects = mock_sox
        # print("⚠️ [兼容性补丁] 已 Mock torchaudio.sox_effects 以支持 Windows 环境。")

    # ================= Torchaudio Load 修复 =================
    # 强制使用 soundfile 加载，绕过 torchcodec 问题
    import soundfile
    import torch
    
    def custom_torchaudio_load(filepath, **kwargs):
        # print(f"Using custom load for {filepath}")
        # soundfile 读取返回 (frames, channels) 或 (frames,)
        data, samplerate = soundfile.read(filepath)
        tensor = torch.from_numpy(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0) # (1, frames)
        else:
            tensor = tensor.t() # (channels, frames)
        return tensor, samplerate
        
    torchaudio.load = custom_torchaudio_load
    # print("⚠️ [兼容性补丁] 已替换 torchaudio.load 以强制使用 soundfile。")

except ImportError:
    pass

# ================= Silero VAD 修复 =================
# 修复 Windows 下 silero_vad 模型文件缺失导致的 RuntimeError
# 由于我们只使用 extract_embedding 而不需要 wespeaker 内置的 VAD，
# 我们可以安全地禁用它，避免加载失败。
try:
    import silero_vad
    # 创建一个假的 VAD 模型对象，防止调用报错
    mock_vad_model = MagicMock()
    # 如果被调用，返回空列表或假的时间戳 (视具体 API 而定，但通常 extract_embedding 不会调用它)
    mock_vad_model.return_value = [] 
    
    # 替换加载函数
    silero_vad.load_silero_vad = MagicMock(return_value=mock_vad_model)
    # print("⚠️ [兼容性补丁] 已禁用 wespeaker 的内置 VAD 加载 (Mock silero_vad)。")
except ImportError:
    pass
# ===================================================

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

DEFAULT_PROXY_DIR = "clips_16k"     # (16k)
DEFAULT_HQ_DIR = "clips_HQ"         # (44.1k)
DEFAULT_RESULT_DIR = "final_result" # 结果目录
DEFAULT_EXAMPLES_DIR = "examples"   # 注册音频目录
DEFAULT_MODEL_PATH = 'models'       # 模型文件夹
DEFAULT_SCORE_THRESHOLD = 0.7    # 阈值

# 进程数：默认使用 CPU 核心数减 2 (留点资源给系统)，且限制最大为 8
# 如果你想要火力全开，改成 os.cpu_count()
DEFAULT_WORKER_NUM = min(8, max(1, os.cpu_count() - 2))
# ===========================================

# 全局变量（用于在子进程中共享模型和声纹库）
worker_model = None
worker_speakers = None
worker_config = None

def compute_cosine_similarity(embed1, embed2):
    e1 = embed1.detach().numpy().flatten()
    e2 = embed2.detach().numpy().flatten()
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(e1, e2) / (norm1 * norm2)

def init_worker(model_path, speakers_emb_dict, config):
    """
    子进程初始化函数：
    每个进程启动时，都会运行一次这个函数。
    用来加载模型和声纹库，避免重复传递数据。
    """
    global worker_model, worker_speakers, worker_config
    
    # 每个进程都屏蔽显卡，强制 CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["WESPEAKER_DEVICE"] = "cpu"
    
    # 加载模型
    # print(f"进程 {os.getpid()} 正在加载模型...")
    worker_model = wespeaker.load_model(model_path)
    worker_model.set_device('cpu')
    
    # 接收主进程传来的主角声纹
    worker_speakers = speakers_emb_dict
    worker_config = config

def process_single_file(proxy_path):
    """
    单个文件的处理逻辑 (将在子进程中运行)
    """
    global worker_model, worker_speakers, worker_config
    
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
        threshold = worker_config.get('SCORE_THRESHOLD', 0.7)
        if best_score >= threshold:
            target_folder = best_name
        
        # 5. 寻找并复制 HQ 文件
        # 计算相对路径
        proxy_dir = worker_config['PROXY_DIR']
        hq_dir = worker_config['HQ_DIR']
        result_dir = worker_config['RESULT_DIR']
        
        if proxy_dir in proxy_path:
             rel_path = proxy_path.split(proxy_dir)[1].lstrip(os.sep)
        else:
             rel_path = os.path.basename(proxy_path) # Fallback
             
        hq_source_path = os.path.join(hq_dir, rel_path)
        
        if os.path.exists(hq_source_path):
            parent_folder = os.path.basename(os.path.dirname(proxy_path))
            filename = os.path.basename(proxy_path)
            new_name = f"{parent_folder}_{filename}"
            
            dst_path = os.path.join(result_dir, target_folder, new_name)
            
            # 执行复制
            shutil.copy(hq_source_path, dst_path)
            return True
        else:
            return False # 没找到 HQ 文件

    except Exception:
        return False

def check_and_download_model(model_path):
    """
    检查模型是否存在，如果不存在则自动下载 'english' 模型并保存到 model_path
    """
    # 检查 model_path 是否包含必要的模型文件
    # WeSpeaker 模型通常包含 avg_model.pt 和 config.yaml
    has_model = False
    if os.path.exists(model_path) and os.path.isdir(model_path):
        files = os.listdir(model_path)
        if 'avg_model.pt' in files and 'config.yaml' in files:
            has_model = True
    
    if not has_model:
        print(f"⚠️ 未在 {model_path} 找到模型，正在自动下载 'english' 模型...")
        try:
            # 这会自动下载模型到 ~/.wespeaker/english
            downloaded_model = wespeaker.load_model('english')
            
            # 获取下载后的模型路径 (通常在 ~/.wespeaker/english)
            # 我们可以尝试从 downloaded_model 对象中获取路径，或者直接假设默认路径
            # 但为了简单起见，我们直接让 wespeaker 加载 'english'，
            # 如果用户指定了本地路径但为空，我们提示用户模型已下载到默认位置，
            # 或者我们可以尝试找到它并复制。
            
            # 由于 wespeaker.load_model 返回的是 Speaker 对象，不直接暴露路径。
            # 但我们可以通过 inspect 或者查看 wespeaker 源码知道默认路径。
            # 为了稳健性，如果本地没找到，我们就直接用 'english' 这个名字加载，
            # 而不是强行用 model_path。
            
            print("✅ 模型下载完成。")
            return 'english' # 返回模型名称，让 wespeaker 自己去缓存找
        except Exception as e:
            print(f"❌ 模型下载失败: {e}")
            return model_path # 还是返回原路径，让它报错
    
    return model_path

def run_identification(proxy_dir, hq_dir, result_dir, examples_dir, model_path, threshold, worker_num):
    print(f"🔥 启动多进程加速 (使用 {worker_num} 个 CPU 核心)...")
    
    # 检查并下载模型
    actual_model_path = check_and_download_model(model_path)

    # 1. 在主进程加载一次模型，为了注册主角
    print(f"正在主进程注册主角声纹 (使用模型: {actual_model_path})...")
    temp_model = wespeaker.load_model(actual_model_path)
    temp_model.set_device('cpu')
    
    speakers_emb = {}
    if not os.path.exists(examples_dir):
        print(f"❌ 错误: 找不到 {examples_dir}")
        return "错误: 找不到样本目录"

    for file in os.listdir(examples_dir):
        if file.lower().endswith('.wav'):
            name = os.path.splitext(file)[0]
            path = os.path.join(examples_dir, file)
            emb = temp_model.extract_embedding(path)
            if np.linalg.norm(emb.detach().numpy()) > 0.1:
                speakers_emb[name] = emb
                print(f"  - 已注册: {name}")
    
    del temp_model # 释放主进程模型，节省内存
    
    if not speakers_emb:
        print("❌ 没有注册样本！")
        return "错误: 没有注册样本"

    # 准备目录
    os.makedirs(os.path.join(result_dir, "Unknown"), exist_ok=True)
    for name in speakers_emb.keys():
        os.makedirs(os.path.join(result_dir, name), exist_ok=True)

    # 2. 扫描文件任务
    print("正在扫描文件列表...")
    task_files = []
    for root, dirs, files in os.walk(proxy_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                task_files.append(os.path.join(root, file))
    
    print(f"共找到 {len(task_files)} 个任务，开始并行处理...")

    # 配置字典，传递给子进程
    config = {
        'PROXY_DIR': proxy_dir,
        'HQ_DIR': hq_dir,
        'RESULT_DIR': result_dir,
        'SCORE_THRESHOLD': threshold
    }

    # 3. 启动进程池
    # initargs 负责把声纹库传给每个子进程
    with ProcessPoolExecutor(max_workers=worker_num, 
                             initializer=init_worker, 
                             initargs=(actual_model_path, speakers_emb, config)) as executor:
        
        # 提交所有任务
        futures = [executor.submit(process_single_file, f) for f in task_files]
        
        success_count = 0
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                success_count += 1

    print(f"\n🎉 全部完成！已处理并归类 {success_count} 个文件到 {result_dir}")
    return f"识别完成，已归类 {success_count} 个文件。"

def main():
    run_identification(DEFAULT_PROXY_DIR, DEFAULT_HQ_DIR, DEFAULT_RESULT_DIR, 
                       DEFAULT_EXAMPLES_DIR, DEFAULT_MODEL_PATH, DEFAULT_SCORE_THRESHOLD, DEFAULT_WORKER_NUM)

if __name__ == "__main__":
    # Windows/WSL 下必须加这行
    multiprocessing.set_start_method('spawn', force=True)
    main()
