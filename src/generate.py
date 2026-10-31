from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

wavs, sample_rate = model.generate_custom_voice(
    text="This is a local text-to-speech inference test.",
    language="English",
    speaker="Ryan",
)

output = Path("outputs/sample.wav")
output.parent.mkdir(parents=True, exist_ok=True)
sf.write(output, wavs[0], sample_rate)
print(f"Saved: {output}")