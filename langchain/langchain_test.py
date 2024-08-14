from langchain_model import Qwen2_LLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from typing import Any
from config import base_config


# 定义一个简单的回调类
class SimpleCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        print(f"Starting LLM run with prompts: {prompts}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        print(f"LLM run completed. Response: {response}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"Error during LLM run: {error}")


# 创建回调实例
callback_handler = SimpleCallbackHandler()

# 创建回调管理器
callback_manager = CallbackManager([callback_handler])

# !!!需要按实际模型位置进行修改
model_name_or_path = base_config.model_name_or_path
llm = Qwen2_LLM(model_name_or_path)
# langchain会调用子类的_call函数
# response = llm("你是谁", callbacks=callback_manager)

# 携带参数
param = {"temperature": 0.5, "max_new_tokens": 550}
response = llm("你是谁", **param)
print(response)
