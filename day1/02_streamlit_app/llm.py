import os
import time
import torch
import streamlit as st
from transformers import pipeline
from huggingface_hub import login

from config import MODEL_NAME

@st.cache_resource
def load_model():
    """LLM モデル（Llama‑3 など）をロードしてキャッシュ"""
    try:
        # Hugging Face トークンを読み込み & ログイン
        hf_token = st.secrets["huggingface"]["token"]
        if hf_token:
            login(hf_token)

        # デバイスと dtype を判定
        device_available = torch.cuda.is_available()
        dtype = torch.float16 if device_available else torch.float32
        st.info(f"Using device: {'cuda' if device_available else 'cpu'}")

        # pipeline を構築
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,     
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe

    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPU メモリ不足、またはトークン未設定の可能性があります。")
        return None


def generate_response(
    pipe,
    user_question: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
):
    """LLM で応答を生成し、履歴を考慮して応答とレイテンシを返す"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0.0

    try:
        start_time = time.time()

        # --- 過去の履歴（直近3ターン）をプロンプトに含める ---
        history = st.session_state.get("chat_history", [])
        context_prompt = ""
        for turn in history[-3:]:  # 直近3ターンのみ使用
            context_prompt += f"ユーザー: {turn['question']}\n"
            context_prompt += f"アシスタント: {turn['answer']}\n"

        # --- 現在の質問を追記 ---
        context_prompt += f"ユーザー: {user_question}\nアシスタント:"

        # --- 推論 ---
        outputs = pipe(
            context_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        # 出力から回答を抽出
        full_text = outputs[0]["generated_text"]
        assistant_response = full_text[len(context_prompt):].strip()

        latency = time.time() - start_time
        return assistant_response, latency

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        return f"エラーが発生しました: {str(e)}", 0.0
