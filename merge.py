import os
import argparse
from pydub import AudioSegment

def merge_wavs(input_folder, output_filename, silence_duration_sec=0.5):
    # 检查文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 '{input_folder}' 不存在。")
        return None

    # 获取所有 .wav 文件
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    
    # 按文件名排序 (这点很重要，否则拼接顺序可能是乱的)
    wav_files.sort()

    if not wav_files:
        print(f"警告: 在 '{input_folder}' 中没有找到 .wav 文件。")
        return None

    print(f"找到 {len(wav_files)} 个 wav 文件，准备开始拼接...")

    # 创建一个 0.5 秒 (500毫秒) 的静音片段
    # pydub 的时间单位是毫秒
    silence = AudioSegment.silent(duration=int(silence_duration_sec * 1000))

    # 初始化空的音频段
    combined_audio = AudioSegment.empty()

    for index, filename in enumerate(wav_files):
        file_path = os.path.join(input_folder, filename)
        print(f"正在处理: {filename}")
        
        try:
            # 读取 wav 文件
            audio_segment = AudioSegment.from_wav(file_path)
            
            # 添加到总音频中
            combined_audio += audio_segment
            
            # 如果不是最后一个文件，在后面加上静音
            if index < len(wav_files) - 1:
                combined_audio += silence
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # 导出结果
    print(f"正在保存结果到: {output_filename} ...")
    combined_audio.export(output_filename, format="wav")
    print("完成！")
    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将文件夹内的所有wav文件拼接成一个，中间包含静音。")
    
    # 命令行参数配置
    parser.add_argument("folder", help="包含wav文件的文件夹路径")
    parser.add_argument("--output", "-o", default="merged_output.wav", help="输出文件名 (默认: merged_output.wav)")
    parser.add_argument("--gap", "-g", type=float, default=0.5, help="间隔静音时长，单位秒 (默认: 0.5)")

    args = parser.parse_args()

    merge_wavs(args.folder, args.output, args.gap)
