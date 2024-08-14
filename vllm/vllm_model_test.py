from vllm_model import call, get_model, get_completion
from config import base_config


# 加载模型
model_name_or_path = base_config.model_name_or_path
llm = get_model(model_name_or_path)

# 多样性文本参数控制
param = dict(max_tokens=512, temperature=0.7)
sampling_params = get_completion(**param)

# 用户提示词
prompt = "介绍你自己？"
response = call(
    llm,
    prompt,
    sampling_params,
)
print(response)
