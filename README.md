# Speaker Recognition WebUI

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

This is a speaker recognition and audio extraction tool based on WeSpeaker and Gradio. It provides a simple web interface to cut audio from videos, generate proxy audio, identify speakers using voiceprint recognition, and merge audio clips for specific speakers.

**Note:** This repository currently contains the source code and scripts. It is not a standalone portable package. You need to configure the Python environment and install dependencies manually, or download the pre-configured portable package (link below).

### Portable Package Download

If you don't want to configure the environment manually, you can download the all-in-one portable package (includes Python environment, ffmpeg, and models).

- **Download Link:** [Insert Link Here]

### Features

*   **Audio Cutting:** Automatically cut audio segments from video files based on subtitle timestamps (.ass, .srt, etc.).
*   **Proxy Generation:** Convert high-quality audio to 16k mono format required by the AI model, with multi-process acceleration.
*   **Speaker Identification:** Use the WeSpeaker ResNet model to identify speakers. Upload a reference audio sample, and the tool will automatically classify audio clips.
*   **Merge & Export:** Merge all audio clips belonging to a specific speaker into a single file.
*   **Windows Optimization:** Includes compatibility patches for `torchaudio` and `silero_vad` on Windows.

### Installation & Usage

#### Prerequisites

*   Windows 10/11 (WSL is also supported but this guide focuses on Windows)
*   Python 3.10+
*   [ffmpeg](https://ffmpeg.org/download.html) (Must be added to PATH or placed in the root directory)
*   [Git](https://git-scm.com/)

#### Manual Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Xiantaidu/Speaker-Recognition-WebUI.git
    cd Speaker-Recognition-WebUI
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    # Install PyTorch (CPU version recommended for smaller size)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install other dependencies
    pip install -r requirements.txt
    ```

4.  Download Models:
    The WeSpeaker model will be automatically downloaded on the first run. You can also manually place the model files in the `models/` directory.

#### Running

*   **Method 1 (Recommended):** Double-click `start.bat`. It will automatically set up the environment and launch the WebUI.
*   **Method 2:** Run via command line:
    ```bash
    python app.py
    ```

Open your browser and visit `http://localhost:7860`.

---

<a name="chinese"></a>
## 中文

这是一个基于 WeSpeaker 和 Gradio 构建的说话人识别与音频提取工具。它提供了一个简单的 Web 界面，用于从视频中截取音频、生成代理音频、使用声纹识别技术识别说话人，并将特定说话人的音频片段合并导出。

**注意：** 本仓库目前包含源代码和脚本，并非独立的便携包。您需要手动配置 Python 环境并安装依赖，或者下载下方提供的预配置便携包。

### 便携包下载

如果您不想手动配置环境，可以下载整合了 Python 环境、ffmpeg 和模型的一键启动便携包。

- **下载链接：** [移动云盘 提取码:aiu7](https://yun.139.com/shareweb/#/w/i/2sxQjMjzkz23u)  [OneDrive 提取码:aYx2](https://savef-my.sharepoint.com/:u:/g/personal/yueyechezu_savef_onmicrosoft_com/IQDgM-CnmGlCRrXqiHHk7O6iAT8rBwhK9eIBaob1HQzfLxM?e=2hnRVm)

### 主要功能

*   **素材截取 (Cut)**：根据字幕时间轴（.ass, .srt 等）自动从视频文件中截取音频片段。
*   **建立代理 (Proxy)**：将高音质音频转换为 AI 模型所需的 16k 单声道格式，支持多进程加速处理。
*   **说话人识别 (Identify)**：使用 WeSpeaker ResNet 模型进行说话人识别。只需上传一段参考音频，工具即可自动将素材归类。
*   **合并导出 (Merge)**：将识别出的某个说话人的所有音频片段合并为一个长音频文件。
*   **Windows 优化**：内置了针对 Windows 环境下 `torchaudio` 和 `silero_vad` 的兼容性修复。

### 安装与使用

#### 前置要求

*   Windows 10/11
*   Python 3.10+
*   [ffmpeg](https://ffmpeg.org/download.html) (需添加到系统 PATH 或直接放在项目根目录)
*   [Git](https://git-scm.com/)

#### 手动安装

1.  克隆仓库：
    ```bash
    git clone https://github.com/Xiantaidu/Speaker-Recognition-WebUI.git
    cd Speaker-Recognition-WebUI
    ```

2.  创建虚拟环境（可选但推荐）：
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```

3.  安装依赖：
    ```bash
    # 安装 PyTorch (推荐 CPU 版本以减小体积)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    # 安装其他依赖
    pip install -r requirements.txt
    ```

4.  下载模型：
    WeSpeaker 模型将在首次运行时自动下载。您也可以手动将模型文件放入 `models/` 目录(推荐)。

#### 运行

*   **方式 1 (推荐)**：双击根目录下的 `start.bat`。它会自动设置环境变量并启动 WebUI。
*   **方式 2**：通过命令行运行：
    ```bash
    python app.py
    ```

启动后，浏览器将自动打开 `http://localhost:7860`。

### 目录结构说明

*   `datasets/`: 默认的素材存放目录（可修改）。
*   `clips_HQ/`: 截取的高音质音频片段。
*   `clips_16k/`: 转换后的 16k 单声道代理音频（用于识别）。
*   `examples/`: 存放说话人参考样本的目录。
*   `final_result/`: 最终识别结果和合并后的音频。
*   `models/`: 存放 WeSpeaker 模型文件。
