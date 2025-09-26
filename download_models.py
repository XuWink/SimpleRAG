from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    'BAAI/bge-large-zh-v1.5',
    cache_dir='models',
    revision='master'  # 或 'main'
)

snapshot_download(
    'deepseek-ai/deepseek-llm-7b-chat',
    cache_dir='models',
    revision='master'  # 或 'main'
)