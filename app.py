import contextlib
import io
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

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
    "responder": {
        "model": os.getenv("RESPONDER_MODEL"),
        "azure_deployment": os.getenv("CHAT_AZURE_DEPLOYMENT"),
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
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

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
    return DuckDuckGoSearchRun().invoke


# =========================
# ヘルパ: 最近の最小限の会話履歴を作成
# =========================
def get_recent_turn_messages(max_turns: int = 2) -> List[Any]:
    """
    セッション履歴(st.session_state.history)から直近の user/assistant のやり取りを
    HumanMessage / AIMessage のリストで返す。各 invoke に渡す最小限のコンテキストとして利用する。
    """
    messages: List[Any] = []
    hist = st.session_state.get("history", [])
    if not hist:
        return messages

    recent = hist[-max_turns:]
    for turn in recent:
        u = turn.get("user_input")
        if u:
            messages.append(HumanMessage(content=u))
        a = turn.get("answer")
        if a:
            messages.append(AIMessage(content=a))
    return messages


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
            "一般的なQA、考察、要約、ガイドは run_code=False とする。"
        )
    )

    # 最小限の履歴（直近の turn を最大2つまで）を渡す
    recent_msgs = get_recent_turn_messages(max_turns=2)
    messages = [system] + recent_msgs + [HumanMessage(content=state["user_input"])]

    decision = llm.with_structured_output(RouterDecision).invoke(messages)
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


class Plan(BaseModel):
    """ユーザ要求を満たすための具体的な処理手順"""

    steps: List[str] = Field(
        description="Pythonで実行する具体的な処理を短い文で表現",
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
            "分解した各命令に対し、それぞれ独立した.pyファイルを作成して実行結果を取得することになります。"
            "外部ネットワークやファイル書込は行わないでください。"
        )
    )

    # planner に渡す履歴は非常に短く（過去1ターン程度）
    recent_msgs = get_recent_turn_messages(max_turns=1)
    decision: Plan = llm.with_structured_output(Plan).invoke(
        [system] + recent_msgs + [HumanMessage(content=state["user_input"])]
    )

    steps = decision.steps
    log = {"agent": "Planner", "output": "\n".join(f"- {s}" for s in steps)}

    return {**state, "plan": steps, "agent_logs": state["agent_logs"] + [log]}


def _extract_code_blocks(text: str) -> List[str]:
    """```python ... ``` のコードブロックを抽出。なければ全文を1ブロック扱い"""
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return [text.strip()] if text.strip() else []


def executor_node(
    state: AppState, ui_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> AppState:
    """実行者：各ステップをPythonコード化し、/tmp/ にファイルを作って実行

    ui_callback が渡された場合、各ステップの実行レポートを即座にUIに渡す。
    """
    llm = get_llm("executor", temperature=0.0)
    executions: List[str] = []

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

        # executor に渡す履歴: これまでの実行結果（あれば）と簡潔な会話履歴
        recent_msgs = get_recent_turn_messages(max_turns=1)
        previous_execs_text = "\n\n".join(executions) if executions else ""
        extra_msgs: List[Any] = []
        if previous_execs_text:
            # 前段の実行ログを渡し、次のステップでの整合性を保つ
            extra_msgs.append(HumanMessage(content=f"これまでの実行ログ:\n{previous_execs_text}"))

        # 最小限のメッセージ列を作成
        prompt_msgs = prompt.format_messages(step=step)
        messages = (
            [SystemMessage(content=prompt_msgs[0].content)]
            if prompt_msgs and isinstance(prompt_msgs[0], SystemMessage)
            else []
        )
        messages = (
            [
                SystemMessage(
                    content=(
                        "あなたはPythonコーダです。以下の『処理ステップ』を満たす最小の実行可能なPythonコードを"
                        "1つのコードブロックだけで出力してください。表示は print を用いてください。"
                        "外部ネットワークやファイルの書き込みはしないでください。"
                    )
                )
            ]
            + recent_msgs
            + extra_msgs
            + [HumanMessage(content=step)]
        )

        code_text = llm.invoke(messages).content
        code_blocks = _extract_code_blocks(code_text)

        if not code_blocks:
            exec_report = f"[Step {i}] コード生成に失敗しました。"
            executions.append(exec_report)
            if ui_callback:
                ui_callback({"agent": f"Executor-Step-{i}", "output": exec_report})
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

        # 各ステップ完了時にUIに即時通知
        if ui_callback:
            ui_callback({"agent": f"Executor-Step-{i}", "output": exec_report})

    log = {"agent": "Executor", "output": "\n\n".join(executions)}
    return {**state, "executions": executions, "agent_logs": state["agent_logs"] + [log]}


def summarizer_node(state: AppState) -> AppState:
    """実行結果の要約"""
    llm = get_llm("summarizer", temperature=0.3)
    joined = "\n\n".join(state.get("executions") or [])

    system = SystemMessage(
        content=(
            "あなたはサマライザです。以下の実行ログを読み、ユーザの要求に対する結果を簡潔に日本語でまとめてください。"
            "必要なら実行値を引用してください。"
        )
    )

    # 要約には、ユーザの元の要求を念のため渡す（文脈確保）
    recent_msgs = get_recent_turn_messages(max_turns=1)
    messages = (
        [system]
        + recent_msgs
        + [
            HumanMessage(content=joined),
            HumanMessage(content=f"元の要求: {state.get('user_input','')}"),
        ]
    )

    # NOTE: もしjoinedが長大になる場合は要注意（本デモでは短いはず）
    answer = llm.invoke(messages).content
    log = {"agent": "Summarizer", "output": answer}
    return {**state, "answer": answer, "agent_logs": state["agent_logs"] + [log]}


def responder_node(state: AppState) -> AppState:
    """
    - chat モード: ユーザの要求（state['user_input']）を直接受けて回答を作成
    - code モード: summarizer の要約（state['answer']）を基に回答を作成
    モードによってシステムプロンプトと渡す内容を切り替え、生成部分は共通化する。
    """
    mode = state.get("mode", "chat")
    generation_llm = get_llm("responder", temperature=0.3)

    if mode == "chat":
        # chat 用プロンプト（シンプルなアシスタント）
        system = SystemMessage(
            content=(
                "あなたは有能なアシスタントです。ユーザの質問に対して、"
                "与えられた情報（直近の会話コンテキストを参考に）を元に、日本語で分かりやすく簡潔に回答してください。"
                "必要なら箇条書きや具体例を用いて説明してください。"
                "出典提示は不要です。"
            )
        )

        # ユーザ発話を直接渡す
        payload = state.get("user_input", "").strip()
        recent_msgs = get_recent_turn_messages(max_turns=3)

    else:
        # code モード（summarizer -> responder）：要約を整形して最終回答にする
        system = SystemMessage(
            content=(
                "あなたは回答者です。以下の要約を基に、ユーザにわかりやすく自然な日本語で最終回答を作成してください。"
                "余計なメタ情報（実行ステップの詳細等）は含めず、必要な部分だけ簡潔に伝えてください。"
            )
        )

        # summarizer の出力を渡す（state['answer'] に要約が入っている想定）
        payload = state.get("answer", "").strip()
        recent_msgs = get_recent_turn_messages(max_turns=1)

    # 共通の生成ステップ
    # system + recent context + human payload の順で渡す
    messages = [system] + recent_msgs + [HumanMessage(content=payload or "")]

    final_answer = generation_llm.invoke(messages).content

    log = {"agent": f"Responder(mode={mode})", "output": final_answer}
    return {**state, "answer": final_answer, "agent_logs": state["agent_logs"] + [log]}


# =========================
# グラフ構築
# =========================
def build_graph():
    g = StateGraph(AppState)

    g.add_node("router", router_node)
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
            "chat": "responder",
            "code": "planner",
        },
    )

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


# UI ヘルパ: エージェントログを即時表示する
def display_agent_log(container: st.delta_generator.DeltaGenerator, log: Dict[str, Any]):
    with container.expander(f"🧩 {log.get('agent')}"):
        # 出力が JSON-ish な場合は st.json を使うと見やい
        out = log.get("output")
        notes = log.get("notes")
        if isinstance(out, (dict, list)):
            st.json(out)
        else:
            st.write(out)
        if notes:
            st.caption("notes:")
            st.json(notes)


# ステップ実行関数: 各ノードが終わるたびにUIに表示
def run_graph_stepwise(
    init_state: AppState, ui_container: st.delta_generator.DeltaGenerator
) -> AppState:
    state = init_state

    # 1) Router
    state = router_node(state)
    # 最新ログを UI に表示
    display_agent_log(ui_container, state["agent_logs"][-1])

    if state["mode"] == "chat":
        # Chat モードでは responder を直接呼ぶ（search を内包）
        state = responder_node(state)
        display_agent_log(ui_container, state["agent_logs"][-1])
        # 回答があれば即時表示
        if state.get("answer"):
            ui_container.success(state["answer"])
        return state

    # Code mode: Planner -> Executor (per-step) -> Summarizer -> Responder
    state = planner_node(state)
    display_agent_log(ui_container, state["agent_logs"][-1])

    # Executor: 各ステップ完了時に即時 UI へ流すコールバックを渡す
    def executor_ui_callback(log: Dict[str, Any]):
        # 一つずつエクスパンダを作って内容を出す
        display_agent_log(ui_container, log)

    state = executor_node(state, ui_callback=executor_ui_callback)
    # Executor 全体のログ（まとめ）も表示
    display_agent_log(ui_container, state["agent_logs"][-1])

    state = summarizer_node(state)
    display_agent_log(ui_container, state["agent_logs"][-1])

    state = responder_node(state)
    display_agent_log(ui_container, state["agent_logs"][-1])

    if state.get("answer"):
        ui_container.success(state["answer"])

    return state


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

    # UI 表示領域（ここに各エージェントの expander を順次追加する）
    ui_container = st.container()

    # ステップ実行（各ノード完了時に即時表示される）
    result_state: AppState = run_graph_stepwise(init_state, ui_container)

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
    "通常会話と判断すると、直接 Responder が検索（必要時）を行って応答します。"
)
