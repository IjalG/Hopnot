"""Knot —— TUI 对话演示。

功能：
- 保存最近 3 轮对话作为短期上下文（滑动窗口）
- 长期记忆使用Knot自动提取+存储
- 可选 LLM API 调用（OpenAI 兼容接口）

配置（环境变量或 .env 文件）：
  BASE_URL  - API 地址（默认 https://api.openai.com/v1）
  API_KEY   - API 密钥
  MODEL_NAME - 模型名（默认 gpt-4o-mini）

用法：
  cd demo
  pip install -r requirements.txt
  cp .env.example .env   # 填入 API_KEY
  python demo.py

不带 API 也能运行（本地模式，仅展示记忆系统）。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── 添加项目根目录到路径 ──────────────────────────────────────────
_demo_dir = Path(__file__).parent
_root_dir = _demo_dir.parent
sys.path.insert(0, str(_root_dir))


# ── 配置 ──────────────────────────────────────────────────────────

@dataclass
class DemoConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    recall_threshold: float = 0.50
    merge_threshold: float = 0.92

    @classmethod
    def from_env(cls) -> "DemoConfig":
        """从环境变量或 .env 文件加载配置。"""
        env_path = _demo_dir / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                val = val.strip().strip("'").strip('"')
                if val:
                    os.environ.setdefault(key.strip(), val)

        return cls(
            base_url=os.environ.get("BASE_URL", cls.base_url),
            api_key=os.environ.get("API_KEY", cls.api_key),
            model_name=os.environ.get("MODEL_NAME", cls.model_name),
        )


# ── 颜色工具 ──────────────────────────────────────────────────────

class Colors:
    """终端 ANSI 颜色。"""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def cyan(s: str) -> str:
        return f"{Colors.CYAN}{s}{Colors.RESET}"

    @staticmethod
    def green(s: str) -> str:
        return f"{Colors.GREEN}{s}{Colors.RESET}"

    @staticmethod
    def yellow(s: str) -> str:
        return f"{Colors.YELLOW}{s}{Colors.RESET}"

    @staticmethod
    def blue(s: str) -> str:
        return f"{Colors.BLUE}{s}{Colors.RESET}"

    @staticmethod
    def gray(s: str) -> str:
        return f"{Colors.GRAY}{s}{Colors.RESET}"

    @staticmethod
    def bold(s: str) -> str:
        return f"{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def dim(s: str) -> str:
        return f"{Colors.DIM}{s}{Colors.RESET}"


# ── 短期上下文（滑动窗口） ──────────────────────────────────────

@dataclass
class Turn:
    """一轮对话。"""
    role: str  # "user" | "assistant"
    content: str


class ShortTermMemory:
    """短期记忆 —— 保存最近 N 轮对话。"""

    def __init__(self, max_turns: int = 3) -> None:
        self._max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self._max_turns:
            self._turns.pop(0)

    def format(self) -> str:
        """格式化为 LLM 可用的上下文。"""
        lines = []
        for turn in self._turns:
            prefix = "用户" if turn.role == "user" else "助手"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)


# ── LLM API 客户端 ───────────────────────────────────────────────

class LLMClient:
    """OpenAI 兼容 API 客户端。"""

    def __init__(self, config: DemoConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model_name
        # 检查 API Key 是否有效（不是占位符）
        placeholders = ["sk-your-api-key-here", "your-api-key", "", "none", "null"]
        self._has_api = bool(self.api_key) and self.api_key.lower() not in placeholders
        # 至少有 8 位才可能是真实 key
        if self._has_api and len(self.api_key) < 8:
            self._has_api = False

    @property
    def available(self) -> bool:
        return self._has_api

    def chat(self, system_prompt: str, user_message: str) -> Optional[str]:
        """调用 LLM API 获取回复。"""
        if not self._has_api:
            return None

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 401:
                print(f"{Colors.RED}[API 错误] 401 认证失败。检查以下项:{Colors.RESET}")
                print(f"  {Colors.YELLOW}1. API Key 是否正确？(前6位: {self.api_key[:6]}...){Colors.RESET}")
                print(f"  {Colors.YELLOW}2. BASE_URL 是否正确？{self.base_url}{Colors.RESET}")
                print(f"  {Colors.YELLOW}3. .env 中值不要加引号{Colors.RESET}")
                return None
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print(f"{Colors.RED}[API 错误] 请求超时(>60s){Colors.RESET}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"{Colors.RED}[API 错误] 无法连接 {self.base_url}{Colors.RESET}")
            return None
        except Exception as e:
            print(f"{Colors.RED}[API 错误] {e}{Colors.RESET}")
            return None


# ── 三元组提取 ───────────────────────────────────────────────────

def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """从文本中提取 <main_point>主体, 关系, 客体</main_point> 三元组。"""
    triples: list[tuple[str, str, str]] = []

    # 格式: <main_point>主体, 关系, 客体</main_point>
    pattern = r"<main_point>\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^<]+?)\s*</main_point>"
    for match in re.finditer(pattern, text):
        s, r, o = match.groups()
        triples.append((s.strip(), r.strip(), o.strip()))

    return triples


# ── 系统 Prompt 模板 ──────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """你是一个知识助手。你的任务是在对话中自然地提取和巩固知识。

## 输出规范

在你的回复末尾，用以下格式输出本次对话中涉及的三元组知识（此内容为系统提取，不要为用户感知）：

<main_point>主体, 关系, 客体</main_point>

例如：
- <main_point>孔明, 是, 蜀汉丞相</main_point>
- <main_point>Python, 用于, 数据分析</main_point>

请确保：
1. 三元组必须准确反映对话中的事实
2. 每个三元组占一行
3. 如果本次对话没有新知识，可以不输出三元组
4. 回复内容保持自然流畅
5. 使用中文回复

## 短期上下文（最近对话）
{recent_context}

## 长期记忆上下文
{memory_context}"""


def get_prompt_template() -> str:
    """获取 Prompt 模板。优先从外部文件加载，否则使用内置模板。"""
    template_path = os.environ.get("PROMPT_TEMPLATE_FILE")
    if template_path:
        tpath = Path(template_path)
        if tpath.is_file():
            return tpath.read_text(encoding="utf-8")
    return SYSTEM_PROMPT_TEMPLATE


# ── 主应用 ────────────────────────────────────────────────────────

class DemoApp:
    """TUI 对话演示主程序。"""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.short_term = ShortTermMemory(max_turns=3)

        # 初始化 LLM 客户端
        self.llm = LLMClient(config)

        # 初始化Knot
        sys.stdout.write(f"{Colors.DIM}Loading memory system...{Colors.RESET}")
        sys.stdout.flush()
        t0 = time.time()

        from knot import HippocampusMemorySystem, get_default_config
        from knot.embedding import Qwen3Embedding, DummyEmbedding

        mem_cfg = get_default_config()
        mem_cfg.recall_threshold = config.recall_threshold
        mem_cfg.merge_threshold = config.merge_threshold

        # 尝试加载 Qwen3，失败则回退到 Dummy
        try:
            embedder = Qwen3Embedding(device="cpu")
            self._embedding_name = "Qwen3-Embedding(1024d)"
        except Exception:
            embedder = DummyEmbedding(dim=64, seed=42)
            self._embedding_name = "DummyEmbedding(64d)"

        self.system = HippocampusMemorySystem(embedding=embedder, config=mem_cfg)

        elapsed = time.time() - t0
        print(f"\r{Colors.GREEN}✓ Memory system loaded{Colors.RESET} "
              f"({self._embedding_name}, {elapsed:.1f}s)")

    # ── 命令处理 ──────────────────────────────────────────────────

    def cmd_help(self) -> None:
        print(Colors.BOLD + "\n  可用命令:" + Colors.RESET)
        print(f"  {Colors.GREEN}/memory{Colors.RESET}   查看长期记忆节点和边")
        print(f"  {Colors.GREEN}/stats{Colors.RESET}    查看系统统计")
        print(f"  {Colors.GREEN}/clear{Colors.RESET}   清空短期上下文")
        print(f"  {Colors.GREEN}/help{Colors.RESET}    显示此帮助")
        print(f"  {Colors.GREEN}/quit{Colors.RESET}    退出")
        print(f"  {Colors.GREEN}/dump{Colors.RESET}    导出记忆快照")
        print()

    def cmd_memory(self) -> None:
        nodes = self.system.graph.get_all_nodes()
        title = f"  📚 长期记忆 ({len(nodes)} 节点, {self.system.graph.edge_count()} 边)"
        print(Colors.BOLD + title + Colors.RESET)
        if not nodes:
            print(Colors.DIM + "  (暂无记忆)" + Colors.RESET)
            return

        for node in sorted(nodes, key=lambda n: n.l3, reverse=True):
            edge_strs = []
            for tid, edge in self.system.graph.get_out_edges(node.id)[:5]:
                tn = self.system.graph.get_node(tid)
                tname = tn.name if tn else tid[:8]
                edge_strs.append(
                    f"→ {tname} ({Colors.YELLOW}{edge.l2:.2f}{Colors.RESET}"
                    + (f" {Colors.DIM}{edge.edge_type.value}{Colors.RESET}" if edge.edge_type.value != "ASSOC" else "")
                    + ")"
                )
            edges_info = " | ".join(edge_strs) if edge_strs else Colors.DIM + "(无出边)" + Colors.RESET
            print(f"  {Colors.GREEN}{node.name}{Colors.RESET}"
                  f" {Colors.DIM}L3={node.l3:.2f} freq={node.freq}{Colors.RESET}")
            print(f"    {edges_info}")
        print()

    def cmd_stats(self) -> None:
        stats = self.system.get_stats()
        print(Colors.BOLD + "\n  📊 系统统计" + Colors.RESET)
        for k, v in stats.items():
            label = k.replace("_", " ").title()
            print(f"  {Colors.DIM}{label}:{Colors.RESET} {Colors.CYAN}{v}{Colors.RESET}")
        print()

    def cmd_dump(self) -> None:
        snapshot = self.system.graph.to_snapshot()
        dump_path = _demo_dir / f"memory_snapshot_{int(time.time())}.json"
        with open(dump_path, "w") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        print(f"  {Colors.GREEN}✓{Colors.RESET} 快照已保存: {dump_path}")
        print()

    # ── 核心逻辑 ──────────────────────────────────────────────────

    def retrieve_memories(self, query: str) -> str:
        """检索长期记忆。"""
        result = self.system.retrieve(query)

        if result.cold_start:
            node = self.system.graph.get_node(result.cold_start_node_id)
            name = node.name if node else "?"
            return (
                f"[发现新知识点] 查询「{query}」是全新内容（节点: {name}），"
                "请尽可能详尽地回答，以便拆解为知识存储。"
            )

        if not result.activated_nodes:
            return "(长期记忆中暂无相关内容)"

        lines = ["[相关长期记忆]"]
        for nid, act in result.activated_nodes:
            node = self.system.graph.get_node(nid)
            if node is None:
                continue
            edges = self.system.graph.get_out_edges(nid)
            facts = []
            for tid, edge in edges[:3]:
                tn = self.system.graph.get_node(tid)
                if tn:
                    facts.append(f"{node.name} → {tn.name}")
            if facts:
                lines.append(f"  · {'; '.join(facts)}")
            else:
                lines.append(f"  · {node.name}")
        return "\n".join(lines)

    def consolidate_response(self, query: str, response: str) -> None:
        """将回复中的三元组整理到记忆。"""
        triples = extract_triples(response)
        if not triples:
            return
        triple_text = "\n".join(f"({s}, {r}, {o})" for s, r, o in triples)
        stats = self.system.consolidate(triple_text, query)
        if stats.get("edges_created", 0) > 0:
            msg = f"📝 提取 {len(triples)} 个三元组, 新建 {stats['edges_created']} 条边"
            ts = time.strftime("%H:%M:%S")
            print(f"  {Colors.DIM}{msg}{Colors.RESET}{Colors.GRAY} [{ts}]{Colors.RESET}")

    def process_user_input(self, user_input: str) -> None:
        """处理用户输入：检索 → LLM → 整理。"""
        # 1. 短期上下文
        self.short_term.add("user", user_input)

        # 2. 检索长期记忆
        memory_context = self.retrieve_memories(user_input)

        # 3. 显示记忆上下文
        if memory_context and not memory_context.startswith("(长期记忆中"):
            mem_lines = memory_context.replace("\n", "\n  ")
            print(f"  {Colors.DIM}🧠 {mem_lines}{Colors.RESET}")

        # 4. 构建 Prompt
        recent_context = self.short_term.format()
        system_prompt = get_prompt_template().format(
            recent_context=recent_context,
            memory_context=memory_context,
        )

        # 5. 调用 LLM
        if self.llm.available:
            sys.stdout.write(f"  {Colors.DIM}⏳ 思考中...{Colors.RESET}")
            sys.stdout.flush()

            response = self.llm.chat(system_prompt, user_input)
            if response:
                # 去掉 <main_point> 标签内容用于显示
                display = re.sub(
                    r"<main_point>\s*[^<]+?\s*</main_point>\s*",
                    "", response
                ).strip()
                if not display:
                    display = response

                print(f"\r{Colors.GREEN}{display}{Colors.RESET}")
                self.short_term.add("assistant", display)
                self.consolidate_response(user_input, response)
                return

        # 6. 本地模式（无 API）
        print(f"\r{Colors.YELLOW}[本地模式] 记忆已记录。配置 .env 后获取 AI 回复。{Colors.RESET}")
        mock = f"（你提到了「{user_input}」，已存入长期记忆。）"
        print(f"{Colors.GREEN}{mock}{Colors.RESET}")
        self.short_term.add("assistant", mock)

    # ── 主循环 ──────────────────────────────────────────────────

    def run(self) -> None:
        """运行主循环。"""
        # 标题
        print()
        print(Colors.BOLD + " ╔══════════════════════════════════════════╗" + Colors.RESET)
        print(Colors.BOLD + " ║    Knot · 对话演示 v1.7     ║" + Colors.RESET)
        print(Colors.BOLD + " ╚══════════════════════════════════════════╝" + Colors.RESET)
        print()
        print(f"  {Colors.DIM}嵌入模型:{Colors.RESET} {Colors.CYAN}{self._embedding_name}{Colors.RESET}")
        model_str = self.llm.model if self.llm.available else "【未配置】本地模式"
        print(f"  {Colors.DIM}LLM 后端:{Colors.RESET} {Colors.CYAN}{model_str}{Colors.RESET}")
        print(f"  {Colors.DIM}短期记忆:{Colors.RESET} {Colors.CYAN}最近 3 轮对话{Colors.RESET}")
        ns = self.system.graph.node_count()
        es = self.system.graph.edge_count()
        print(f"  {Colors.DIM}长期记忆:{Colors.RESET} {Colors.CYAN}记忆图 ({ns} 节点 / {es} 边){Colors.RESET}")
        print()
        print(Colors.DIM + "  命令: /memory  /stats  /clear  /help  /quit" + Colors.RESET)
        print()

        # 无 API 时注入示例知识
        if not self.llm.available:
            self.system.consolidate(
                "(Knot, 是, 纯图结构记忆模拟)\n"
                "(检索阶段, 包含, 种子选取)\n"
                "(检索阶段, 包含, 随机游走扩散)\n"
                "(检索阶段, 包含, 输出截断)\n"
                "(整理阶段, 包含, 节点定位)\n"
                "(整理阶段, 包含, 边处理决策树)\n"
                "(整理阶段, 包含, 偏置漂移)\n"
                "(整理阶段, 包含, 三角闭合)",
                "系统初始化",
            )
            print(f"  {Colors.DIM}📚 已注入示例知识（8 个三元组）{Colors.RESET}")
            print()

        while True:
            try:
                user_input = input(f"{Colors.BLUE}你{Colors.RESET} {Colors.DIM}> {Colors.RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd == "/quit":
                break
            elif cmd == "/help":
                self.cmd_help()
            elif cmd == "/memory":
                self.cmd_memory()
            elif cmd == "/stats":
                self.cmd_stats()
            elif cmd == "/clear":
                self.short_term.clear()
                print(f"  {Colors.GREEN}✓{Colors.RESET} 短期上下文已清空\n")
            elif cmd == "/dump":
                self.cmd_dump()
            elif user_input.startswith("/"):
                print(f"  {Colors.RED}未知命令:{Colors.RESET} {user_input}  (输入 /help 查看可用命令)\n")
            else:
                self.process_user_input(user_input)
                print()

        print(f"\n{Colors.GREEN}再见！{Colors.RESET}\n")


def main() -> None:
    config = DemoConfig.from_env()
    app = DemoApp(config)
    app.run()


if __name__ == "__main__":
    main()
