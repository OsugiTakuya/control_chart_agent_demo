import contextlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

# ツール（Web検索 & Python実行）
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# LangGraph
from langgraph.graph import END, StateGraph

# --- LangChain / OpenAI ---
from pydantic import BaseModel, Field

# SerpAPI を使いたい場合はコメントアウト解除（.envにSERPAPI_API_KEYが必要）
# from langchain_community.tools import SerpAPIWrapper


# =========================
# Env 読み込み & LLM Factory
# =========================
load_dotenv()


# 各エージェント用のモデル/デプロイ設定を.envから読み込み
AGENT_CONFIG = {
    "router": {
        "model": os.getenv("ROUTER_MODEL"),
        "azure_deployment": os.getenv("ROUTER_AZURE_DEPLOYMENT"),
    },
    "chat": {
        "model": os.getenv("CHAT_MODEL"),
        "azure_deployment": os.getenv("CHAT_AZURE_DEPLOYMENT"),
    },
    "planner": {
        "model": os.getenv("PLANNER_MODEL"),
        "azure_deployment": os.getenv("PLANNER_AZURE_DEPLOYMENT"),
    },
    "executor": {
        "model": os.getenv("EXECUTOR_MODEL"),
        "azure_deployment": os.getenv("EXECUTOR_AZURE_DEPLOYMENT"),
    },
    "summarizer": {
        "model": os.getenv("SUMMARIZER_MODEL"),
        "azure_deployment": os.getenv("SUMMARIZER_AZURE_DEPLOYMENT"),
    },
}


def get_llm(agent: str, temperature: float = 0.2):
    """
    agent名に応じた LLM インスタンスを返す。
    - Azure 環境変数が揃っていれば Azure を使用（デプロイ名はエージェントごとに切替）
    - それ以外は OpenAI API を使用
    """
    cfg = AGENT_CONFIG[agent]
    model = cfg["model"]
    azure_deployment = cfg["azure_deployment"]

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if azure_endpoint and azure_key and azure_deployment:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
            deployment_name=azure_deployment,
            temperature=temperature,
        )

    # 通常の OpenAI API
    return ChatOpenAI(model=model, temperature=temperature)


# =========================
# Web検索ツールの用意
# =========================
def get_web_search_tool():
    # SERPAPI_API_KEY があれば SerpAPI を使ってもよいが、ここでは DuckDuckGo をデフォルト採用
    # serp_key = os.getenv("SERPAPI_API_KEY")
    # if serp_key:
    #     serp = SerpAPIWrapper(serpapi_api_key=serp_key)
    #     return serp.run
    return DuckDuckGoSearchRun().invoke


# =========================
# ルータ出力 (Structured)
# =========================
class RouterDecision(BaseModel):
    """ユーザ指示が 'コードを実行して結果を返す' 種かどうかの分類"""

    run_code: bool = Field(
        description="True if the user requests writing AND executing code to produce outputs; False for normal conversation / explanation."
    )
    reason: str = Field(description="Short reason for the decision in Japanese.")


# =========================
# LangGraph の State 定義
# =========================
class AppState(TypedDict):
    user_input: str
    mode: Literal["chat", "code"]
    plan: Optional[List[str]]
    executions: Optional[List[str]]
    answer: Optional[str]
    agent_logs: List[Dict[str, Any]]  # UI用：各エージェントのログ


# =========================
# ノード実装
# =========================


def router_node(state: AppState) -> AppState:
    """ユーザ指示が通常会話か、コード実行が必要かを判定"""
    llm = get_llm("router", temperature=0.0)
    system = SystemMessage(
        content=(
            "あなたは入力が『通常の会話か/コードを書いて実行して結果を返すべきか』を判定する分類器です。"
            "データ加工・計算・グラフ作成・シミュレーションなど、明らかに実行結果が必要な場合は run_code=True。"
            "一般的なQA、考察、要約、ガイド、Web検索で簡潔するものは run_code=False とする。"
        )
    )
    human = HumanMessage(content=state["user_input"])
    decision = llm.with_structured_output(RouterDecision).invoke([system, human])
    mode: Literal["chat", "code"] = "code" if decision.run_code else "chat"
    log = {"agent": "Router", "output": f"run_code={decision.run_code} / reason={decision.reason}"}
    return {
        **state,
        "mode": mode,
        "agent_logs": state["agent_logs"] + [log],
    }


class SearchQueries(BaseModel):
    """検索クエリ候補。検索不要な場合は空リストにする。"""

    queries: List[str] = Field(
        description="DuckDuckGoで検索する短いクエリ。検索不要なら空リスト。",
        min_items=0,
        max_items=3,
    )


def chat_agent_node(state: AppState) -> AppState:
    """通常会話＋Web検索ツールを使うエージェント"""
    llm = get_llm("chat", temperature=0.3)
    search = get_web_search_tool()

    # 検索クエリ生成（structured output）
    planner_llm = get_llm("chat", temperature=0.0)
    system = SystemMessage(
        content=(
            "あなたはリサーチャです。ユーザの質問に答えるために必要なら検索クエリを考えます。"
            "DuckDuckGoで有効な短い検索クエリを生成してください。"
            "検索不要と判断した場合は空リストを返してください。"
        )
    )
    human = HumanMessage(content=state["user_input"])
    decision: SearchQueries = planner_llm.with_structured_output(SearchQueries).invoke(
        [system, human]
    )

    queries = decision.queries
    results: List[str] = []

    if queries:
        for q in queries:
            try:
                res = search(q)
                results.append(f"[{q}]\n{res}")
            except Exception as e:
                results.append(f"[{q}] 検索エラー: {e}")

    # 最終回答
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは有能なリサーチャです。以下の検索結果（空のこともあります）を参考に、"
                "日本語で簡潔に回答してください。必要なら箇条書きを使ってください。",
            ),
            ("human", "{q}"),
            ("ai", "{snippets}"),
        ]
    )
    answer = llm.invoke(
        prompt.format_messages(
            q=state["user_input"],
            snippets="\n\n".join(results) if results else "（検索なし）",
        )
    ).content

    log = {
        "agent": "ChatAgent(with WebSearch)",
        "output": answer,
        "notes": {"queries": queries, "search_snippets": results},
    }
    return {**state, "answer": answer, "agent_logs": state["agent_logs"] + [log]}


class Plan(BaseModel):
    """ユーザ要求を満たすための具体的な処理手順"""

    steps: List[str] = Field(
        description="Pythonで実行する具体的な処理ステップを短い文で表現",
        min_items=1,
        max_items=3,
    )


def planner_node(state: AppState) -> AppState:
    """コード実行が必要な場合の計画立案（structured output版）"""
    llm = get_llm("planner", temperature=0.2)

    system = SystemMessage(
        content=(
            "あなたはプランナーです。ユーザ要求を満たすための実行計画を立てます。"
            "計画は必ず Python で実行可能な具体的な処理ステップとして、"
            "短い文のリストに分解してください。"
            "外部ネットワークやファイル書込は行わないでください。"
        )
    )
    human = HumanMessage(content=state["user_input"])

    # structured_outputを使って steps を直接抽出
    decision: Plan = llm.with_structured_output(Plan).invoke([system, human])

    steps = decision.steps
    log = {"agent": "Planner", "output": "\n".join(f"- {s}" for s in steps)}

    return {**state, "plan": steps, "agent_logs": state["agent_logs"] + [log]}


def _extract_code_blocks(text: str) -> List[str]:
    """```python ... ``` のコードブロックを抽出。なければ全文を1ブロック扱い"""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return [text.strip()] if text.strip() else []


def executor_node(state: AppState) -> AppState:
    """実行者：各ステップをPythonコード化し、/tmp/ にファイルを作って実行"""
    llm = get_llm("executor", temperature=0.0)
    executions: List[str] = []

    tmp_dir = Path("tmp")

    for i, step in enumerate(state.get("plan", []) or [], start=1):
        # ステップを Python コード化
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはPythonコーダです。以下の『処理ステップ』を満たす最小の実行可能なPythonコードを"
                    "1つのコードブロックだけで出力してください。表示は print を用いてください。"
                    "外部ネットワークやファイルの書き込みはしないでください。",
                ),
                ("human", "{step}"),
            ]
        )
        code_text = llm.invoke(prompt.format_messages(step=step)).content
        code_blocks = _extract_code_blocks(code_text)

        if not code_blocks:
            executions.append(f"[Step {i}] コード生成に失敗しました。")
            continue

        code = code_blocks[0]

        # ファイル書き出し
        file_path = tmp_dir / f"step_{i}.py"
        file_path.write_text(code, encoding="utf-8")

        # サブプロセスで実行
        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            out = result.stdout.strip() or "(no output)"
            err = result.stderr.strip()
            exec_report = f"[Step {i}] OK\n--- file ---\n{file_path}\n--- code ---\n{code}\n--- output ---\n{out}"
            if err:
                exec_report += f"\n--- stderr ---\n{err}"
        except Exception as e:
            exec_report = (
                f"[Step {i}] 実行エラー: {e}\n--- file ---\n{file_path}\n--- code ---\n{code}"
            )

        executions.append(exec_report)

    log = {"agent": "Executor", "output": "\n\n".join(executions)}
    return {**state, "executions": executions, "agent_logs": state["agent_logs"] + [log]}


def summarizer_node(state: AppState) -> AppState:
    """実行結果の要約"""
    llm = get_llm("summarizer", temperature=0.3)
    joined = "\n\n".join(state.get("executions") or [])
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはサマライザです。以下の実行ログを読み、ユーザの要求に対する結果を簡潔に日本語でまとめてください。"
                "必要なら実行値を引用してください。",
            ),
            ("human", "{log}"),
        ]
    )
    answer = llm.invoke(prompt.format_messages(log=joined)).content
    log = {"agent": "Summarizer", "output": answer}
    return {**state, "answer": answer, "agent_logs": state["agent_logs"] + [log]}


def responder_node(state: AppState) -> AppState:
    """Summarizer の要約をもとに、ユーザ向けの最終回答を自然な文章に整形"""
    llm = get_llm("summarizer", temperature=0.5)  # summarizerと同じでも良いが切替可
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは回答者です。以下の要約を基に、ユーザにわかりやすく自然な日本語で最終回答を作成してください。"
                "余計なメタ情報（実行ステップなど）は含めず、必要な部分だけ簡潔に伝えてください。",
            ),
            ("human", "{summary}"),
        ]
    )
    answer = llm.invoke(prompt.format_messages(summary=state.get("answer", ""))).content
    log = {"agent": "Responder", "output": answer}
    return {**state, "answer": answer, "agent_logs": state["agent_logs"] + [log]}


# =========================
# グラフ構築
# =========================
def build_graph():
    g = StateGraph(AppState)

    g.add_node("router", router_node)
    g.add_node("chat_agent", chat_agent_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("summarizer", summarizer_node)
    g.add_node("responder", responder_node)

    g.set_entry_point("router")

    def route_decision(state: AppState):
        return state["mode"]

    g.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat": "chat_agent",
            "code": "planner",
        },
    )

    g.add_edge("chat_agent", END)
    g.add_edge("planner", "executor")
    g.add_edge("executor", "summarizer")
    g.add_edge("summarizer", "responder")
    g.add_edge("responder", END)

    return g.compile()


GRAPH = build_graph()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="LLM Multi-Agent Demo", page_icon="🤖", layout="centered")
st.title("🤖 LLM Multi-Agent Demo (LangGraph + LangChain + Streamlit)")

# セッション永続メモリ
if "history" not in st.session_state:
    st.session_state.history = []  # 各ターンの state を保存
if "thread_memory" not in st.session_state:
    # ユーザ発話を単純に連結で持つ（必要に応じて高度なメモリに置換可）
    st.session_state.thread_memory = []

with st.form(key="user_form", clear_on_submit=False):
    user_input = st.text_area(
        "指示文を入力してください",
        height=160,
        placeholder="例：2020〜2024の世界GDP上位5か国を一覧にして考察して\n例：πを100桁計算して先頭20桁だけ表示して",
    )
    submitted = st.form_submit_button("送信")

if submitted and user_input.strip():
    # スレッド記憶を付加（本デモではシンプルに Human の履歴を渡すだけ）
    st.session_state.thread_memory.append(user_input.strip())

    init_state: AppState = {
        "user_input": user_input.strip(),
        "mode": "chat",
        "plan": None,
        "executions": None,
        "answer": None,
        "agent_logs": [],
    }

    # LangGraph 実行
    result_state: AppState = GRAPH.invoke(init_state)

    # 履歴に保存（セッションの記憶保持）
    st.session_state.history.append(result_state)

# 履歴の表示
if st.session_state.history:
    st.subheader("📚 セッション履歴")
    for idx, turn in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"### Turn {len(st.session_state.history) - idx + 1}")
        for log in turn.get("agent_logs", []):
            with st.expander(f"🧩 {log.get('agent')}"):
                st.write(log.get("output"))
                if "notes" in log and log["notes"]:
                    st.caption("notes:")
                    st.json(log["notes"])
        if turn.get("answer"):
            st.success(turn["answer"])
else:
    st.info("最初の指示を入力して、エージェントの実行結果を確認してください。")


st.caption(
    "Tips: ルータがコード実行と判断すると、Planner → Executor → Summarizer の順で動きます。"
    "通常会話と判断すると、Web検索付きの ChatAgent が応答します。"
)
