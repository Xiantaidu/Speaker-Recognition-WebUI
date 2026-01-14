import gradio as gr
import os
import shutil
import subprocess
from cut_batch import run_cut_batch
from make_proxy import run_make_proxy
from identify import run_identification
from merge import merge_wavs

# ================= 配置 =================
BASE_DIR = os.getcwd()
SOURCE_DIR = os.path.join(BASE_DIR, "bocchi_the_rock")
OUTPUT_ROOT = os.path.join(BASE_DIR, "clips_HQ")
PROXY_DIR = os.path.join(BASE_DIR, "clips_16k")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
RESULT_DIR = os.path.join(BASE_DIR, "final_result")
MODEL_PATH = os.path.join(BASE_DIR, "models")

# 确保目录存在
for d in [SOURCE_DIR, OUTPUT_ROOT, PROXY_DIR, EXAMPLES_DIR, RESULT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def step1_cut(video_files, sub_files, clean_old):
    if not video_files:
        return "错误: 请上传视频文件。"
    if not sub_files:
        return "错误: 请上传字幕文件。"
    
    # 1. 清理 SOURCE_DIR
    if clean_old and os.path.exists(SOURCE_DIR):
        print("正在清理旧文件...")
        for f in os.listdir(SOURCE_DIR):
            try:
                os.remove(os.path.join(SOURCE_DIR, f))
            except Exception as e:
                print(f"清理文件失败: {e}")
    
    # 2. 复制文件
    print("正在复制文件到工作目录...")
    for v_path in video_files:
        # Gradio 上传的文件路径通常包含原始文件名
        dest_name = os.path.basename(v_path)
        shutil.copy(v_path, os.path.join(SOURCE_DIR, dest_name))
        
    for s_path in sub_files:
        dest_name = os.path.basename(s_path)
        shutil.copy(s_path, os.path.join(SOURCE_DIR, dest_name))
        
    # 3. 运行切分
    return run_cut_batch(SOURCE_DIR, OUTPUT_ROOT)

def step2_proxy():
    return run_make_proxy(OUTPUT_ROOT, PROXY_DIR)

def step3_identify(ref_files, threshold, clean_old, num_workers):
    if not ref_files:
        return "错误: 请上传至少一个说话人样本。"
    
    # 1. 准备样本目录
    if clean_old:
        if os.path.exists(EXAMPLES_DIR):
            shutil.rmtree(EXAMPLES_DIR)
        os.makedirs(EXAMPLES_DIR)
    
    # 2. 保存样本并转换为 16k 单声道
    names = []
    print("正在处理样本音频...")
    for f_path in ref_files:
        filename = os.path.basename(f_path)
        dest_path = os.path.join(EXAMPLES_DIR, filename)
        
        # 使用 ffmpeg 转换音频: -ar 16000 (16k采样率) -ac 1 (单声道) -y (覆盖)
        try:
            subprocess.run([
                'ffmpeg', '-i', f_path, 
                '-ar', '16000', 
                '-ac', '1', 
                '-y', dest_path
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  - 已转换并保存: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"  - 转换失败 {filename}: {e}")
            # 如果转换失败，尝试直接复制（虽然可能导致识别报错）
            shutil.copy(f_path, dest_path)

        names.append(os.path.splitext(filename)[0])
    
    print(f"已注册说话人: {names}")
    
    # 3. 运行识别
    return run_identification(PROXY_DIR, OUTPUT_ROOT, RESULT_DIR, EXAMPLES_DIR, MODEL_PATH, threshold, int(num_workers))

def refresh_speakers():
    if not os.path.exists(RESULT_DIR):
        return gr.update(choices=[])
    
    speakers = []
    for d in os.listdir(RESULT_DIR):
        d_path = os.path.join(RESULT_DIR, d)
        if os.path.isdir(d_path):
            # 检查里面是否有 wav 文件
            wavs = [f for f in os.listdir(d_path) if f.endswith('.wav')]
            if wavs:
                speakers.append(d)
    
    return gr.update(choices=sorted(speakers))

def step4_merge(speaker_name):
    if not speaker_name:
        return None
    
    speaker_dir = os.path.join(RESULT_DIR, speaker_name)
    output_filename = os.path.join(RESULT_DIR, f"{speaker_name}_merged.wav")
    
    result_path = merge_wavs(speaker_dir, output_filename)
    return result_path

# ================= 界面构建 =================
theme = gr.themes.Soft()

with gr.Blocks(title="Speaker Recognition WebUI", theme=theme) as demo:
    gr.Markdown("# 🎙️ 说话人识别与提取工具")
    
    with gr.Tab("1. 素材截取 (Cut)"):
        gr.Markdown("### 第一步：上传视频和同名字幕")
        gr.Markdown("""
        **说明**：此步骤将根据字幕时间轴从视频中截取音频片段。
        
        **支持格式**：
        *   视频：`.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.wav`
        *   字幕：`.ass`, `.srt`, `.ssa`, `.vtt`
        
        **操作指南**：
        1.  上传视频文件和对应的字幕文件。
        2.  **重要**：请确保视频和字幕的文件名（不含后缀）完全一致（例如 `ep01.mp4` 和 `ep01.ass`）。
        3.  勾选“处理前清理旧素材文件”以清空之前上传的文件（推荐）。
        4.  点击“开始截取”按钮。
        """)
        with gr.Row():
            vid_input = gr.File(label="上传视频文件", file_count="multiple")
            sub_input = gr.File(label="上传字幕文件", file_count="multiple")
        
        clean_source_chk = gr.Checkbox(label="处理前清理旧素材文件", value=True)
        
        cut_btn = gr.Button("开始截取", variant="primary")
        cut_output = gr.Textbox(label="运行日志")
        cut_btn.click(step1_cut, [vid_input, sub_input, clean_source_chk], cut_output)

    with gr.Tab("2. 建立代理 (Proxy)"):
        gr.Markdown("### 第二步：生成 AI 专用音频")
        gr.Markdown("""
        **说明**：此步骤将截取的高音质音频转换为 AI 模型所需的格式（16k 采样率，单声道）。
        
        **操作指南**：
        1.  确保第一步已成功完成。
        2.  点击“开始转换”按钮。
        """)
        proxy_btn = gr.Button("开始转换", variant="primary")
        proxy_output = gr.Textbox(label="运行日志")
        proxy_btn.click(step2_proxy, [], proxy_output)

    with gr.Tab("3. 说话人识别 (Identify)"):
        gr.Markdown("### 第三步：自定义说话人识别")
        gr.Markdown("""
        **说明**：此步骤使用声纹识别模型，将音频片段归类到不同的说话人文件夹中。
        
        **操作指南**：
        1.  上传目标说话人的参考音频（样本）。
        2.  **重要**：文件名将作为说话人的名字。例如上传 `bocchi.wav`，识别出的片段将放入 `bocchi` 文件夹。
        3.  上传几个文件就识别几个人。
        4.  调整相似度阈值（默认 0.7）和 CPU 线程数。
        5.  点击“开始识别”。
        """)
        with gr.Row():
            ref_input = gr.File(label="上传样本音频 (.wav)", file_count="multiple")
            with gr.Column():
                threshold = gr.Slider(0.0, 1.0, value=0.7, label="相似度阈值 (越高越严格)")
                num_workers = gr.Slider(1, os.cpu_count(), value=max(1, os.cpu_count() - 2), step=1, label="CPU 线程数 (并行处理)")
                clean_examples_chk = gr.Checkbox(label="识别前清理旧样本", value=True)
                
        id_btn = gr.Button("开始识别", variant="primary")
        id_output = gr.Textbox(label="运行日志")
        id_btn.click(step3_identify, [ref_input, threshold, clean_examples_chk, num_workers], id_output)

    with gr.Tab("4. 合并导出 (Merge)"):
        gr.Markdown("### 第四步：结果合并与导出")
        gr.Markdown("""
        **说明**：此步骤将识别出的某个说话人的所有音频片段合并为一个长音频。
        
        **操作指南**：
        1.  点击“刷新列表”以加载最新的识别结果。
        2.  在下拉菜单中选择一个说话人。
        3.  点击“合并下载”生成音频文件。
        """)
        with gr.Row():
            speaker_select = gr.Dropdown(label="选择说话人", choices=[])
            refresh_btn = gr.Button("🔄 刷新列表")
        merge_btn = gr.Button("合并下载", variant="primary")
        audio_result = gr.Audio(label="合并后的音频")
        
        refresh_btn.click(refresh_speakers, outputs=speaker_select)
        merge_btn.click(step4_merge, inputs=speaker_select, outputs=audio_result)

if __name__ == "__main__":
    # 允许在局域网访问，并自动打开浏览器
    demo.launch(server_name="0.0.0.0", inbrowser=True)
