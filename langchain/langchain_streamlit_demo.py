from langchain_model import Qwen2_LLM
import streamlit as st
import datetime
import pytz
import time

model_name_or_path = "/root/autodl-tmp/insult_code/qwen/Qwen2-7B-Instruct"


@st.cache_resource
def load_model():
    llm = Qwen2_LLM(model_name_or_path)
    return llm


def main():
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False
    if st.session_state.first_visit:
        # 获取当前的 UTC 时间
        utc_now = datetime.datetime.now(tz=pytz.utc)
        # 将 UTC 时间转换为北京时间
        beijing_tz = pytz.timezone("Asia/Shanghai")
        beijing_now = utc_now.astimezone(beijing_tz)

        # 设置 session state 中的 date_time 为北京时间
        st.session_state.date_time = beijing_now
        # st.balloons()  # 弹出气球
    # 加载模型
    llm = load_model()
    # 左侧导航栏
    with st.sidebar:
        st.sidebar.header("Streamlit 构建 LLM 模型demo")
        st.caption(":fire: [Github代码地址](https://github.com/hgsw/llm.git)")
        # 导航栏显示日期
        st.sidebar.date_input("当前日期：", st.session_state.date_time.date())

        st.sidebar.subheader("我可以帮你解决以下问题：")
        background = "我是一个人工智能助手，可以陪你解决各种问题， 比如文本生成，文本概括等。"
        st.markdown(background)

        st.sidebar.subheader("参数控制：")
        # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
        max_length = st.slider("Max input tokens", 0, 1024, 512, step=1)
        max_new_tokens = st.slider("Max output tokens", 0, 1024, 512, step=1)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
        # 方框风格的选择样式
        # temperature = st.number_input("Temperature", min_value=0.01, max_value=0.99, value=0.7, step=0.05)

        # 清空会话删除所有历史对话记录
        st.sidebar.subheader("危险操作：")
        st.button("清空会话", on_click=init_session)

    st.title("LLM Chat Robot")
    st.caption("🚀 A streamlit chatbot demo, base langchain inference")

    # 初始化问候消息
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "你好，有什么可以帮助你吗？"})
    # 显示所有历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 提问框和对话模块设置
    if prompt := st.chat_input("Ask anything"):
        prompt = check_len(prompt, max_length)
        st.chat_message("user").markdown(prompt)
        # 保存历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 打印机效果
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # 调用大模型获取结果
            param = dict(max_length=max_length, max_new_tokens=max_new_tokens, temperature=temperature)
            print(f"parameter: {param}")
            response = generate(llm, prompt, **param)
            for trunk in list(response):
                full_response += trunk
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
    # 保证初始化问候语无法删除
    if len(st.session_state.messages) >= 2:
        deleted = st.session_state.messages[-1]
        st.button("删除消息", on_click=delete_message, args={deleted["content"]})


def check_len(prompt, max_length=512):
    if len(prompt) > max_length:
        return f"输入文本超出限制{max_length}，请重新输入~"
    else:
        return prompt


def delete_message(message):
    for msg in st.session_state.messages:
        if msg["content"] == message:
            st.session_state.messages.remove(msg)
            break


def init_session():
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "你好，有什么可以帮助你吗？"})


def generate(llm, prompt, **param):
    response = llm(prompt, **param)
    return response


if __name__ == "__main__":
    main()
