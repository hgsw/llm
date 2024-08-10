from vllm import LLM, SamplingParams


def get_model(model_name_or_path):
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        max_num_seqs=100,
        trust_remote_code=True,
    )
    return llm


def get_completion(tokenizer=None, **kwargs):
    stop_token_ids = [151329, 151336, 151338]
    # SamplingParams部分默认参数和Qwen2-7B-Instruct配置文件的参数不一致
    # 参数配置来自qwen2/Qwen2-7B-Instruct/generation_config.json
    # 不修改无影响，仅仅是为了和langchain推理的参数保持一致
    kwargs["repetition_penalty"] = 1.05
    kwargs["top_p"] = 0.8
    kwargs["top_k"] = 20
    # print(f"SamplingParams param: {kwargs}")
    sampling_params = SamplingParams(**kwargs, stop_token_ids=stop_token_ids)

    return sampling_params


def call(llm, prompt, sampling_params):
    outputs = llm.generate(prompt, sampling_params)
    for output in outputs:
        response = output.outputs[0].text
    return response
