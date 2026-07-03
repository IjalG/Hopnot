# 演示程序

TUI 对话演示，展示 Hopnot 的完整工作流程。

## 快速开始

```bash
cd demo
pip install -r requirements.txt
cp .env.example .env     # 填入 API 信息
python demo.py
```

## 配置

编辑 `.env` 文件：

```ini
BASE_URL=https://api.openai.com/v1
API_KEY=sk-your-key-here
MODEL_NAME=gpt-4o-mini

# 可选：自定义 Prompt 模板文件
# PROMPT_TEMPLATE_FILE=./my_template.txt
```

支持任何 OpenAI 兼容接口，例如 DeepSeek：

```ini
BASE_URL=https://api.deepseek.com/v1
API_KEY=sk-xxx
MODEL_NAME=deepseek-chat
```

## 功能

| 功能 | 说明 |
|:---|:---|
| **短期记忆** | 自动保存最近 3 轮对话，超出丢弃最早 |
| **长期记忆** | LLM 回复中的 `<main_point>` 三元组自动提取存入图 |
| **LLM 调用** | 基于 BASE_URL/API_KEY/MODEL_NAME 调用 AI |
| **本地模式** | 无 API 时自动运行，展示 Hopnot 功能 |
| **冷启动** | 新知识自动创建节点，激活值 = 1.0 |

## 命令

| 命令 | 说明 |
|:---|:---|
| `/memory` | 查看长期记忆图和关联强度 |
| `/stats` | 系统统计信息 |
| `/clear` | 清空短期上下文 |
| `/dump` | 导出记忆快照 |
| `/help` | 帮助 |
| `/quit` | 退出 |

## 工作流

```
用户输入 → 检索长期记忆（检索阶段）
→ 构建 Prompt（系统指令 + 短期3轮 + 长期记忆上下文）
→ 调用 LLM API
→ 解析回复中的 <main_point> 三元组
→ 存入 Hopnot（整理阶段）
```

## 自定义 Prompt 模板

设置 `PROMPT_TEMPLATE_FILE` 环境变量指向自定义模板文件：

```bash
export PROMPT_TEMPLATE_FILE=./my_prompt.txt
python demo.py
```

模板中使用 `{recent_context}` 和 `{memory_context}` 占位符。
