import streamlit as st

# --- 内部モジュール ---
import ui                      # UI コンポーネント
import llm                     # LLM ラッパー
import database                # SQLite ラッパー
import metrics                 # NLTK & 評価指標
import data                    # サンプルデータ挿入

# --- アプリ設定 ---
st.set_page_config(page_title="Llama‑3 Chatbot", layout="wide")

# --- 初期化 ---
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()

# LLM モデルをロード（llm.load_model 内で st.cache_resource）
pipe = llm.load_model()

# --- タイトル & 説明 ---
st.title("Gemma 2 Chatbot with Feedback")
st.write("Gemma 2 Chatbot with Feedbackを使用したチャットボットです。Temperature / top‑p を調整しながら回答をテストし、フィードバックを記録できます。")
st.markdown("---")

# --- サイドバー ナビゲーション ---
if "page" not in st.session_state:
    st.session_state.page = "チャット"

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page),
    key="page_selector",
    on_change=lambda: setattr(st.session_state, "page", st.session_state.page_selector),
)

# --- ページ切り替え ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")

elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()

elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッター ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: Arisa.N")
