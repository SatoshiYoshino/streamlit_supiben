import os

import streamlit as st
from streamlit_chat import message

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Does it work?
from langchain.callbacks.streamlit import StreamlitCallbackHandler

system_message = """
あなたは、質問に対して「GMOインターネットグループスピリットベンチャー宣言」の内容を含めて回答するBotです。\
スピベンの中から質問にあった文章を１つ回答してください。またその後に解説をしてください。
以下の条件を守って質問に対して回答してください。\
[条件]質問の内容に近いGMOインターネットグループスピリットベンチャー宣言の情報の１文を必ず選択して回答をします。\
[条件]【前提】から下の文章を参照する
[条件]GMOインターネットグループスピリットベンチャー宣言の情報には合わない回答はしない\
[条件]GMOインターネットグループスピリットベンチャー宣言=スピベン
[条件]スピベンというキーワードが出たらGMOインターネットグループスピリットベンチャー宣言と理解する
[条件]「GMOインターネットグループスピリットベンチャー宣言」はこのURLを参照するhttps://www.gmo.jp/brand/sv/
[条件]質問に対する答えとして適切なGMOインターネットグループスピリットベンチャー宣言の一文を回答します
  """
prompt = ChatPromptTemplate.from_messages([
  SystemMessagePromptTemplate.from_template(system_message),
  MessagesPlaceholder(variable_name="history"),
  HumanMessagePromptTemplate.from_template("{input}")
])

@st.cache_resource
def load_conversation():
  llm = ChatOpenAI(
    streaming=True,
    callback_manager=CallbackManager([
      StreamlitCallbackHandler(),
      StreamingStdOutCallbackHandler()
    ]),
    verbose=True,
    temperature=0,
    max_tokens=1024
  )
  memory = ConversationBufferMemory(return_messages=True)
  conversation = ConversationChain(
    memory=memory,
    prompt=prompt,
    llm=llm
  )
  return conversation

st.title("スピベン先生")

if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

with st.form("質問する"):
  user_message = st.text_area("質問を入力してください")

  submitted = st.form_submit_button("質問する")
  if submitted:
    conversation = load_conversation()
    answer = conversation.predict(input=user_message)

    st.session_state.past.append(user_message)
    st.session_state.generated.append(answer)

    if st.session_state["generated"]:
      for i in range(len(st.session_state.generated) - 1, -1, -1):
        message(st.session_state.generated[i], key=str(i))
        message(st.session_state.past[i], is_user=True, key=str(i) + "_user")
