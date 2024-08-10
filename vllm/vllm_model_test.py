from vllm_model2 import generate

model_name_or_path = "qwen/Qwen2-7B-Instruct"
# prompt = "你好，帮我介绍一下什么是大语言模型。"
prompt = "介绍你自己？"
param = dict(temperature=0.7, max_tokens=512)
response = generate(
    model_name_or_path=model_name_or_path,
    prompt=prompt,
    **param,
)

print(response)
