import sys
import os
from types import ModuleType

# 添加当前目录到 sys.path
sys.path.append(os.getcwd())

# ================= 注入 torchaudio.sox_effects 补丁 =================
try:
    import torchaudio
    import torch
    if not hasattr(torchaudio, 'sox_effects'):
        fake_sox_effects = ModuleType('torchaudio.sox_effects')
        
        def _fake_apply_effects_tensor(tensor, sample_rate, effects, channels_first=True):
            print("Warning: torchaudio.sox_effects.apply_effects_tensor called but not implemented.")
            return tensor, sample_rate

        def _fake_apply_effects_file(path, effects, normalize=True, channels_first=True, format=None):
            print("Warning: torchaudio.sox_effects.apply_effects_file called but not implemented.")
            return torch.zeros(1, 16000), 16000

        fake_sox_effects.apply_effects_tensor = _fake_apply_effects_tensor
        fake_sox_effects.apply_effects_file = _fake_apply_effects_file
        
        torchaudio.sox_effects = fake_sox_effects
        sys.modules['torchaudio.sox_effects'] = fake_sox_effects
except ImportError:
    pass
# ===================================================================

print("Checking imports...")
try:
    import gradio
    import pysubs2
    import pydub
    import tqdm
    import numpy
    import torch
    import torchaudio
    import requests
    import s3prl
    import wespeaker
    # 尝试导入 app.py 中可能用到的其他模块
    from identify import run_identification
    print("All imports successful!")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
