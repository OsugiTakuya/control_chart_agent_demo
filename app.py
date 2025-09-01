import datetime
import logging
import os
import re
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

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
# ロガー初期化 — ファイル出力（ローテート）
# =========================
# 環境変数でログディレクトリや回転設定を上書き可能
log_dir = Path(os.getenv("LOG_DIR", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "llm_multi_agent.log"

logger = logging.getLogger("llm_multi_agent")
logger.setLevel(logging.INFO)
# Streamlit のリロードでハンドラを重複して追加しないようにする
if not logger.handlers:
    # コンソールハンドラ（従来通り）
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(ch)

    # RotatingFileHandler: maxBytes（デフォルト 5MB）, backupCount（デフォルト 3）
    try:
        max_bytes = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "3"))
        fh = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)
    except Exception as e:
        # ファイルハンドラの作成に失敗してもアプリは続行。コンソールへエラーを出す。
        logger.error("Failed to create RotatingFileHandler: %s", e)

    # ルートロガーへの伝播を止めて二重ログ化を避ける
    logger.propagate = False

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

    return ChatOpenAI(model=model, temperature=temperature)


# =========================
# デフォルトの agent system prompts
# =========================
tool_description = (
    "利用可能なツール:\n"
    "- tools.save_similar_series_plot(filepath: str, target_time: str | datetime.datetime, window: int, product: str, parameter: str) -> None\n"
    "  指定した target_time を含む時系列部分系列と類似した区間を検索し、図を生成して filepath に保存する。\n"
    "  target_time は文字列または datetime、window は直近データ点数、product は製品名、parameter はパラメータ名を指定。\n"
)

DEFAULT_AGENT_PROMPTS = {
    "router": (
        "あなたは入力が『通常の会話か/コードを書いて実行して結果を返すべきか』を判定する分類器です。"
        "データ加工・計算・グラフ作成・シミュレーションなど、実行結果が必要な場合は run_code=True。"
        "一般的なQAや要約、説明は run_code=False。"
    ),
    "planner": (
        "あなたはプランナーです。ユーザ要求を満たすための処理手順を、"
        "実行可能なPythonステップに分解して短いリストで出力してください。\n\n"
        f"{tool_description}\n"
        "- 時系列類似箇所の可視化が目的なら必ず save_similar_series_plot を計画に含めてください。\n"
        "- 各ステップは独立した .py ファイルとして実行されます。\n"
        "- 外部ネットワークは使用しないでください。"
    ),
    "executor_1": (
        "あなたはPythonコーダです。与えられた処理ステップを満たす最小の実行可能コードを、"
        "1つのコードブロックだけで出力してください。出力は print を用いてください。"
        "ツール用モジュール tools は既にインポートされています。\n\n"
        "図を生成する場合のルール:\n"
        "- matplotlib 等で作成した場合、必ず '{image_path_str}' に保存し、"
        "直後に `IMAGE_SAVED: {image_path_str}` を print してください。\n"
        "- 保存後は plt.close() を呼んでください。\n"
        "- 直接の計算やテキスト出力なら通常通り print を使ってください。\n\n"
        f"{tool_description}"
    ),
    "executor_2": "これまでの実行ログ:\n{previous_execs_text}",
    "summarizer_1": (
        "あなたはサマライザです。以下の実行ログを読み、ユーザの要求に対する結果を簡潔に日本語でまとめてください。"
        "必要なら実行値を引用してください。"
    ),
    "summarizer_2": "元の要求: {user_input}",
    "responder_chat": (
        "あなたは有能なアシスタントです。ユーザの質問に対して、"
        "与えられた情報を元に、日本語で分かりやすく簡潔に回答してください。"
        "必要なら箇条書きや具体例を使っても構いません。"
    ),
    "responder_code_1": (
        "あなたは回答者です。ユーザの質問に対して、"
        "与えられた情報を元に日本語で簡潔に回答してください。"
        "余計なメタ情報（実行ステップの詳細やソースコード等）は含めず、"
        "必要な部分だけを伝えてください。"
    ),
    "responder_code_2": "# ユーザの質問:\n{user_input}\n\n# 分析結果:\n{summarizer_answer}",
}


class RouterDecision(BaseModel):
    """Routerエージェントの出力"""

    run_code: bool = Field(
        description="True if the user requests writing AND executing code to produce outputs; False for normal conversation / explanation."
    )
    reason: str = Field(description="Short reason for the decision in Japanese.")


class Plan(BaseModel):
    """Plannerエージェントの出力"""

    steps: List[str] = Field(
        description="Pythonで実行する具体的な処理を短い文で表現",
        min_items=1,
        max_items=3,
    )


# =========================
# セッションプロンプト初期化ヘルパ
# =========================
if "agent_prompts" not in st.session_state:
    st.session_state.agent_prompts = DEFAULT_AGENT_PROMPTS.copy()


# =========================
# Web検索ツールの用意
# =========================
def get_web_search_tool():
    return DuckDuckGoSearchRun().invoke


# =========================
# ヘルパ: 最近の最小限の会話履歴を作成
# =========================
def get_recent_turn_messages(max_turns: int = 2) -> List[Any]:
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


# ログ出力フォーマッタ（Message オブジェクトを見やすくする）
def _format_messages_for_log(msgs: List[Any]) -> str:
    out_lines: List[str] = []
    for m in msgs:
        try:
            role = m.__class__.__name__
            content = getattr(m, "content", str(m))
        except Exception:
            role = "Unknown"
            content = str(m)
        out_lines.append(f"- {role}: {content}")
    return "\n".join(out_lines) if out_lines else "(no messages)"


# ログを統一して出すユーティリティ
def log_agent_io(agent: str, input_data: Any, output_data: Any):
    try:
        if isinstance(input_data, (list, tuple)):
            input_str = _format_messages_for_log(list(input_data))
        else:
            input_str = str(input_data)
    except Exception:
        input_str = repr(input_data)

    try:
        output_str = str(output_data)
    except Exception:
        output_str = repr(output_data)

    logger.info("=== %s INPUT ===\n%s", agent, input_str)
    logger.info("=== %s OUTPUT ===\n%s", agent, output_str)


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
# ノード実装（各所で session の agent_prompts を参照）
# =========================
def router_node(state: AppState) -> AppState:
    llm = get_llm("router", temperature=0.0)
    system_content = st.session_state.agent_prompts.get("router", DEFAULT_AGENT_PROMPTS["router"])
    system = SystemMessage(content=system_content)

    recent_msgs = get_recent_turn_messages(max_turns=2)
    messages = [system] + recent_msgs + [HumanMessage(content=state["user_input"])]

    # ログ出力（input）
    log_agent_io("Router", messages, "invoking LLM for decision")

    decision = llm.with_structured_output(RouterDecision).invoke(messages)

    # ログ出力（output）
    log_agent_io(
        "Router", "(invocation)", f"run_code={decision.run_code} / reason={decision.reason}"
    )

    mode: Literal["chat", "code"] = "code" if decision.run_code else "chat"
    log = {"agent": "Router", "output": f"run_code={decision.run_code} / reason={decision.reason}"}
    return {
        **state,
        "mode": mode,
        "agent_logs": state["agent_logs"] + [log],
    }


class SearchQueries(BaseModel):
    queries: List[str] = Field(min_items=0, max_items=3)


def planner_node(state: AppState) -> AppState:
    llm = get_llm("planner", temperature=0.2)
    system_content = st.session_state.agent_prompts.get("planner", DEFAULT_AGENT_PROMPTS["planner"])
    system = SystemMessage(content=system_content)

    recent_msgs = get_recent_turn_messages(max_turns=1)

    # ログ（input: recent history + user input）
    planner_input_msgs = (
        [system] + recent_msgs + [HumanMessage(content=state.get("user_input", ""))]
    )
    log_agent_io("Planner", planner_input_msgs, "invoking LLM to create plan")

    decision: Plan = llm.with_structured_output(Plan).invoke(
        [system] + recent_msgs + [HumanMessage(content=state["user_input"])]
    )

    steps = decision.steps
    log = {"agent": "Planner", "output": "\n".join(f"- {s}" for s in steps)}

    # ログ（output: steps）
    log_agent_io("Planner", "(invocation)", steps)

    return {**state, "plan": steps, "agent_logs": state["agent_logs"] + [log]}


def _extract_code_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    return [text.strip()] if text.strip() else []


def executor_node(
    state: AppState, ui_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> AppState:
    llm = get_llm("executor", temperature=0.0)
    executions: List[str] = []
    images_accum: List[str] = []

    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(state.get("plan", []) or [], start=1):
        image_path = (tmp_dir / f"step_{i}.png").resolve()
        image_path_str = str(image_path)

        # ユーザ編集の executor プロンプト（ベース）を取得し、画像保存に関する「動的ルール」を追記する
        user_executor_prompt_base = st.session_state.agent_prompts.get(
            "executor_1", DEFAULT_AGENT_PROMPTS["executor_1"]
        )
        system_content = user_executor_prompt_base.format(image_path_str=image_path_str)

        # ChatPromptTemplate を用意（system + human）
        prompt = ChatPromptTemplate.from_messages([("system", system_content), ("human", "{step}")])

        recent_msgs = get_recent_turn_messages(max_turns=1)
        previous_execs_text = "\n\n".join(executions) if executions else ""
        extra_msgs: List[Any] = []
        if previous_execs_text:
            system_content = st.session_state.agent_prompts.get(
                "executor_2", DEFAULT_AGENT_PROMPTS["executor_2"]
            )
            extra_msgs.append(
                HumanMessage(content=system_content.format(previous_execs_text=previous_execs_text))
            )

        prompt_msgs = prompt.format_messages(step=step)
        messages = (
            [SystemMessage(content=prompt_msgs[0].content)]
            if prompt_msgs and isinstance(prompt_msgs[0], SystemMessage)
            else []
        )
        messages = (
            [SystemMessage(content=prompt_msgs[0].content)]
            + recent_msgs
            + extra_msgs
            + [HumanMessage(content=step)]
        )

        # ログ（Executor-step: input: step, recent messages, previous execs）
        executor_input_summary = {
            "step_index": i,
            "step": step,
            "recent_messages": _format_messages_for_log(recent_msgs),
            "previous_execs": previous_execs_text[:4000],
        }
        log_agent_io(f"Executor-Step-{i}", executor_input_summary, "invoking executor LLM")

        code_text = llm.invoke(messages).content

        # ログ（生成コード）
        log_agent_io(f"Executor-Step-{i}", "(invocation)", code_text[:4000])

        code_blocks = _extract_code_blocks(code_text)

        if not code_blocks:
            exec_report = f"[Step {i}] コード生成に失敗しました。"
            executions.append(exec_report)
            if ui_callback:
                ui_callback({"agent": f"Executor-Step-{i}", "output": exec_report})
            continue

        code = code_blocks[0]

        # 先頭に必要なimport文を追記
        code = (
            "import sys\n"
            "import os\n"
            "CURDIR = os.path.dirname(os.path.abspath(__file__))\n"
            "sys.path.append(os.path.join(CURDIR, '..'))\n"
            "from libs import tools\n"
        ) + code

        file_path = tmp_dir / f"step_{i}.py"
        file_path.write_text(code, encoding="utf-8")

        prior_files = set(tmp_dir.iterdir())

        try:
            # 40秒タイムアウト、標準出力・標準エラーをキャプチャ
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=40,
            )
            out = result.stdout.strip() or "(no output)"
            err = result.stderr.strip()
            exec_report_str = f"[Step {i}] OK\n--- file ---\n{file_path}\n--- code ---\n{code}\n--- output ---\n{out}"
            if err:
                exec_report_str += f"\n--- stderr ---\n{err}"
        except Exception as e:
            out = ""
            err = str(e)
            exec_report_str = (
                f"[Step {i}] 実行エラー: {e}\n--- file ---\n{file_path}\n--- code ---\n{code}"
            )

        # ログ（実行結果）
        log_agent_io(f"Executor-Step-{i}", "(execution)", {"stdout": out, "stderr": err})

        images: List[str] = []
        try:
            marker_matches = re.findall(r"IMAGE_SAVED:\s*(\S+)", out)
            for m in marker_matches:
                p = Path(m)
                if not p.is_absolute():
                    p = (tmp_dir / m).resolve()
                if p.exists():
                    images.append(str(p))
                    images_accum.append(str(p))
        except Exception:
            pass

        new_files = set(tmp_dir.iterdir()) - prior_files
        for p in sorted(new_files):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}:
                if str(p) not in images:
                    images.append(str(p))
                    images_accum.append(str(p))

        executions.append(exec_report_str)

        step_log = {"agent": f"Executor-Step-{i}", "output": exec_report_str}
        if images:
            step_log["images"] = images

        if ui_callback:
            ui_callback(step_log)

    log = {"agent": "Executor", "output": "\n\n".join(executions)}
    if images_accum:
        log["images"] = images_accum

    # Executor 全体ログ出力
    log_agent_io("Executor", "(all executions)", executions[:4000])

    return {**state, "executions": executions, "agent_logs": state["agent_logs"] + [log]}


def summarizer_node(state: AppState) -> AppState:
    llm = get_llm("summarizer", temperature=0.3)
    joined = "\n\n".join(state.get("executions") or [])

    # ログ（input: 全実行ログ）
    log_agent_io("Summarizer", joined[:4000], "invoking summarizer")

    system_content = st.session_state.agent_prompts.get(
        "summarizer_1", DEFAULT_AGENT_PROMPTS["summarizer_1"]
    )
    system = SystemMessage(content=system_content)

    recent_msgs = get_recent_turn_messages(max_turns=1)
    user_input = state.get("user_input", "").strip()
    system_content = st.session_state.agent_prompts.get(
        "summarizer_2", DEFAULT_AGENT_PROMPTS["summarizer_2"]
    )
    messages = (
        [system]
        + recent_msgs
        + [
            HumanMessage(content=joined),
            HumanMessage(content=system_content.format(user_input=user_input)),
        ]
    )

    answer = llm.invoke(messages).content

    # ログ（output: summary）
    log_agent_io("Summarizer", "(invocation)", answer)

    log = {"agent": "Summarizer", "output": answer}
    return {**state, "answer": answer, "agent_logs": state["agent_logs"] + [log]}


def responder_node(state: AppState) -> AppState:
    mode = state.get("mode", "chat")
    generation_llm = get_llm("responder", temperature=0.3)

    if mode == "chat":
        system_content = st.session_state.agent_prompts.get(
            "responder_chat", DEFAULT_AGENT_PROMPTS["responder_chat"]
        )
        system = SystemMessage(content=system_content)
        payload = state.get("user_input", "").strip()
        recent_msgs = get_recent_turn_messages(max_turns=3)

        # ログ（input: recent history + payload）
        responder_input = [system] + recent_msgs + [HumanMessage(content=payload)]
        log_agent_io("Responder(chat)", responder_input, "invoking responder LLM")
    else:
        system_content = st.session_state.agent_prompts.get(
            "responder_code_1", DEFAULT_AGENT_PROMPTS["responder_code_1"]
        )
        system = SystemMessage(content=(system_content))

        user_input = state.get("user_input", "").strip()
        summarizer_answer = state.get("answer", "").strip()
        system_content = st.session_state.agent_prompts.get(
            "responder_code_2", DEFAULT_AGENT_PROMPTS["responder_code_2"]
        )
        payload = system_content.format(user_input=user_input, summarizer_answer=summarizer_answer)
        recent_msgs = get_recent_turn_messages(max_turns=1)

        # ログ（input: user_input + summarizer）
        responder_input = [system] + recent_msgs + [HumanMessage(content=payload or "")]
        log_agent_io("Responder(code)", responder_input, "invoking responder LLM")

    messages = [system] + recent_msgs + [HumanMessage(content=payload or "")]
    final_answer = generation_llm.invoke(messages).content

    # ログ（output: final answer）
    log_agent_io("Responder", "(invocation)", final_answer)

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
# Streamlit UI (タブ化)
# =========================
st.set_page_config(page_title="LLM Multi-Agent Demo", page_icon="🤖", layout="centered")
tabs = st.tabs(["アプリ", "詳細設定"])
app_tab, settings_tab = tabs[0], tabs[1]

with app_tab:
    st.title("🤖 LLM Multi-Agent Demo (LangGraph + LangChain + Streamlit)")

    # セッション永続メモリ
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_memory" not in st.session_state:
        st.session_state.thread_memory = []

    # UI ヘルパ: エージェントログを即時表示する
    def display_agent_log(container: st.delta_generator.DeltaGenerator, log: Dict[str, Any]):
        with container.expander(f"🧩 {log.get('agent')}"):
            out = log.get("output")
            notes = log.get("notes")
            if isinstance(out, (dict, list)):
                st.json(out)
            else:
                st.write(out)
            if notes:
                st.caption("notes:")
                st.json(notes)

    def run_graph_stepwise(
        init_state: AppState, ui_container: st.delta_generator.DeltaGenerator
    ) -> AppState:
        state = init_state

        # Router
        state = router_node(state)
        display_agent_log(ui_container, state["agent_logs"][-1])

        if state["mode"] == "chat":
            state = responder_node(state)
            display_agent_log(ui_container, state["agent_logs"][-1])
            if state.get("answer"):
                ui_container.success(state["answer"])
            return state

        # Code mode
        state = planner_node(state)
        display_agent_log(ui_container, state["agent_logs"][-1])

        def executor_ui_callback(log: Dict[str, Any]):
            display_agent_log(ui_container, log)

        state = executor_node(state, ui_callback=executor_ui_callback)
        # Executor 全体ログ表示
        display_agent_log(ui_container, state["agent_logs"][-1])

        state = summarizer_node(state)
        display_agent_log(ui_container, state["agent_logs"][-1])

        state = responder_node(state)
        display_agent_log(ui_container, state["agent_logs"][-1])

        executor_images: List[str] = []
        for log in state.get("agent_logs", []):
            imgs = log.get("images")
            if imgs:
                executor_images.extend(imgs)

        if state.get("answer"):
            ui_container.success(state["answer"])

        if executor_images:
            try:
                ui_container.image(executor_images, use_column_width=True)
            except Exception as e:
                ui_container.warning(f"図の表示に失敗しました: {e}")

        return state

    with st.form(key="user_form", clear_on_submit=False):
        user_input = st.text_area(
            "指示文を入力してください",
            height=160,
            placeholder="例：製品○○、評価値△△の管理図をプロットして\n例：製品○○、評価値△△が異常か判定して",
        )
        submitted = st.form_submit_button("送信")

    if submitted and user_input.strip():
        st.session_state.thread_memory.append(user_input.strip())

        init_state: AppState = {
            "user_input": user_input.strip(),
            "mode": "chat",
            "plan": None,
            "executions": None,
            "answer": None,
            "agent_logs": [],
        }

        ui_container = st.container()
        result_state: AppState = run_graph_stepwise(init_state, ui_container)
        st.session_state.history.append(result_state)

    # 履歴表示
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

with settings_tab:
    st.header("🛠 詳細設定 — 各エージェントの system プロンプトを編集")
    st.write(
        "ここで編集した system プロンプトは、このセッション中に保存され、エージェント実行時に使用されます。"
    )
    st.write(
        "※ Executor の場合、図の保存や `IMAGE_SAVED:` マーカー等の画像ルールは自動的に追記されます。"
    )

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("デフォルトに戻す"):
            st.session_state.agent_prompts = DEFAULT_AGENT_PROMPTS.copy()
            st.rerun()

    # 各エージェントのプロンプト編集エリア
    for agent in DEFAULT_AGENT_PROMPTS.keys():
        label = f"{agent}"
        current = st.session_state.agent_prompts.get(agent, DEFAULT_AGENT_PROMPTS[agent])
        new_text = st.text_area(label, value=current, height=160, key=f"prompt_{agent}")
        # 変更があればセッション状態に反映
        st.session_state.agent_prompts[agent] = new_text

    st.caption("編集後はアプリタブに戻り、通常どおり送信してください。")
