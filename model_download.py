# model_download.py

from modelscope import snapshot_download, AutoModel, AutoTokenizer

# 下载指定模型
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='./', revision='master')