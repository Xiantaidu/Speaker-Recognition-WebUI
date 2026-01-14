import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 配置 =================
DEFAULT_HQ_DIR = "clips_HQ"      # 刚才重新切分的高音质目录
DEFAULT_PROXY_DIR = "clips_16k"  # 给AI准备的低音质替身目录
# =======================================

def process_single_proxy(args):
    src, dst = args
    try:
        # 确保目标子文件夹存在
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # 如果替身已经存在，跳过（节省时间）
        if os.path.exists(dst):
            return False
            
        # 调用 ffmpeg 进行转码
        # -ac 1: 单声道
        # -ar 16000: 16k采样率
        # -vn: 去除视频流(保险)
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', src,
            '-ac', '1', '-ar', '16000', '-vn',
            dst
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Error processing {src}: {e}")
        return False

def run_make_proxy(hq_dir, proxy_dir):
    if not os.path.exists(proxy_dir):
        os.makedirs(proxy_dir)
    
    print("正在生成 AI 专用替身文件 (16k Mono)...")
    
    if not os.path.exists(hq_dir):
        print(f"错误: 找不到 HQ 目录 {hq_dir}")
        return "错误: 找不到 HQ 目录"

    tasks = []
    # 扫描所有 HQ 文件
    for root, dirs, files in os.walk(hq_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                src_path = os.path.join(root, file)
                
                # 计算相对路径，保持目录结构一致
                # 例如: clips_HQ/01/abc.wav -> 01/abc.wav
                rel_path = os.path.relpath(src_path, hq_dir)
                dst_path = os.path.join(proxy_dir, rel_path)
                
                tasks.append((src_path, dst_path))

    # 使用 CPU 核心数 - 1 作为进程数，避免卡死系统，且限制最大为 8
    num_workers = min(8, max(1, os.cpu_count() - 1))
    print(f"启动 {num_workers} 个进程进行并行转换...")

    count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_single_proxy, task) for task in tasks]
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                count += 1

    print("替身生成完毕！")
    return f"处理完成，生成了 {count} 个代理文件。"

def main():
    run_make_proxy(DEFAULT_HQ_DIR, DEFAULT_PROXY_DIR)

if __name__ == "__main__":
    # Windows/WSL 下必须加这行
    multiprocessing.set_start_method('spawn', force=True)
    main()
