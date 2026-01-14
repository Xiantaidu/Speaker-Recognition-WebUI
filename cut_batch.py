import os
import pysubs2
from pydub import AudioSegment
from tqdm import tqdm

# ================= 配置区域 =================
# 默认配置，如果作为脚本运行将使用这些值
DEFAULT_SOURCE_DIR = "bocchi_the_rock"
DEFAULT_OUTPUT_ROOT = "clips_HQ"
DEFAULT_MIN_DURATION_MS = 800
DEFAULT_PADDING_MS = 50

# 支持的视频格式和字幕格式
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wav')
SUB_EXTS = ('.ass', '.srt', '.ssa', '.vtt')
# ===========================================

def is_lyric_style(style_name):
    """
    判断样式名称是否属于歌词
    只要样式名以 OP, ED, 或 IN 开头，即视为歌词
    """
    if not style_name:
        return False
        
    target_prefixes = ['OP', 'ED', 'IN']
    style_name = style_name.upper()
    
    # 检查是否以特定前缀开头 (涵盖 OPJP, EDCN, IN-Romaji 等)
    return any(style_name.startswith(prefix) for prefix in target_prefixes)

def process_single_video(video_path, sub_path, output_folder, min_duration_ms=800, padding_ms=50):
    """处理单个视频的函数"""
    print(f"--> 正在加载音频: {os.path.basename(video_path)}")
    
    try:
        # 加载音频 (保留原音质，不转码，后续由 identify 脚本处理)
        audio = AudioSegment.from_file(video_path)
    except Exception as e:
        print(f"    [错误] 无法加载视频音频: {e}")
        return

    print(f"--> 正在加载字幕: {os.path.basename(sub_path)}")
    try:
        subs = pysubs2.load(sub_path)
    except Exception as e:
        print(f"    [错误] 无法解析字幕: {e}")
        return

    # 确保输出子目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    skipped_lyrics = 0
    
    # 遍历字幕
    for line in tqdm(subs, desc="切分进度", leave=False):
        # ================= 新增：歌词过滤逻辑 =================
        # pysubs2 解析出的 line 对象直接包含 .style 属性
        if is_lyric_style(line.style):
            skipped_lyrics += 1
            continue
        # ======================================================

        start = max(0, line.start - padding_ms)
        end = min(len(audio), line.end + padding_ms)
        duration = end - start

        if duration < min_duration_ms:
            continue

        clip = audio[start:end]
        
        # 文件名: 序号_开始_结束.wav
        filename = f"{count:05d}_{start}_{end}.wav"
        save_path = os.path.join(output_folder, filename)
        
        clip.export(save_path, format="wav")
        count += 1
    
    print(f"    完成！生成了 {count} 个片段 (跳过了 {skipped_lyrics} 句歌词)。")
    return count

def run_cut_batch(source_dir, output_root, min_duration_ms=800, padding_ms=50):
    # 1. 扫描文件夹
    if not os.path.exists(source_dir):
        print(f"错误: 找不到文件夹 {source_dir}")
        return "错误: 找不到源文件夹"

    files = os.listdir(source_dir)
    video_files = [f for f in files if f.lower().endswith(VIDEO_EXTS)]
    
    if not video_files:
        print("该目录下没有找到视频文件。")
        return "未找到视频文件"

    print(f"找到 {len(video_files)} 个视频文件，准备开始处理...")
    
    total_clips = 0

    # 2. 遍历每一个视频文件
    for vid_file in video_files:
        video_path = os.path.join(source_dir, vid_file)
        
        # 获取文件名（不带后缀），例如 "1.mp4" -> "1"
        base_name = os.path.splitext(vid_file)[0]
        
        # 3. 寻找同名字幕文件
        found_sub = None
        for ext in SUB_EXTS:
            sub_candidate = os.path.join(source_dir, base_name + ext)
            if os.path.exists(sub_candidate):
                found_sub = sub_candidate
                break
        
        if found_sub:
            print(f"\n[{base_name}] 匹配成功: 视频 + 字幕")
            # 为每个视频创建一个单独的输出子文件夹，避免文件名冲突
            # 结果存在: clips_all/1/0001.wav, clips_all/2/0001.wav
            video_output_dir = os.path.join(output_root, base_name)
            count = process_single_video(video_path, found_sub, video_output_dir, min_duration_ms, padding_ms)
            if count:
                total_clips += count
        else:
            print(f"\n[{base_name}] 跳过: 未找到同名的 .ass/.srt 字幕文件")

    print("\n所有任务全部完成！")
    return f"处理完成，共生成 {total_clips} 个片段。"

def main():
    run_cut_batch(DEFAULT_SOURCE_DIR, DEFAULT_OUTPUT_ROOT, DEFAULT_MIN_DURATION_MS, DEFAULT_PADDING_MS)

if __name__ == "__main__":
    main()
