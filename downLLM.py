
from huggingface_hub import snapshot_download
import os
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 目标模型名称（Hugging Face Hub 上的模型ID）
model_name = "Qwen/Qwen3-Embedding-8B"  # 替换为正确的模型ID

# 本地保存路径
local_model_path = "./Model/Qwen3-Embedding-8B"

# 如果目录不存在，则创建
os.makedirs(local_model_path, exist_ok=True)

# 下载模型
print(f"Downloading {model_name} to {local_model_path}...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    local_dir_use_symlinks=False,  # 避免使用符号链接
    resume_download=True,          # 支持断点续传
    token=None,                   # 如果需要访问私有模型，提供 Hugging Face Token
)

print(f"Model downloaded to {local_model_path}!")
