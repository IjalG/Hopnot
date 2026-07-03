# Knot

**实验性项目** — 纯图结构的动态记忆模拟系统（检索 + 整理闭环）。

Knot 用有向加权图模拟人脑的记忆读取与巩固，不依赖任何神经网络梯度训练。

> **⚠️ 实验性声明**
>
> 本项目处于早期实验阶段，API 和内部机制可能随实践反馈而调整。
> 欢迎所有人参与改进：调整代码、优化参数、提交 PR、提出 issue。
> 尤其欢迎有 GPU 的大神开发 GPU 加速版本，以提升嵌入计算速度（本人穷，只有 CPU 🙃）。

## 一行起步

```python
from knot import HippocampusMemorySystem

system = HippocampusMemorySystem()
system.consolidate("(Python, 是, 编程语言)", query="初始化")
result = system.retrieve("Python")
```

## 文档

| 文档 | 内容 |
|------|------|
| [技术规格](docs/technical.md) | 架构、公式、双向量 |
| [使用指南](docs/usage.md) | 安装、配置、示例 |
| [API 参考](docs/api-reference.md) | 全部类和方法 |
| [参数调优](docs/tuning.md) | 超参数说明 |
| [演示程序](docs/demo.md) | TUI 对话演示 |
| [English](docs/README.en.md) | English documentation |

## 项目结构

```
knot/                 # 核心库
├── graph.py          # 图结构（节点/边 CRUD）
├── retrieval.py      # 检索阶段（RWR、种子选取）
├── consolidation.py  # 整理阶段（边决策树、偏置漂移）
├── knot/memory_system.py  # 主协调器
├── embedding.py      # 嵌入接口（Qwen3 / Dummy）
├── config.py         # 超参数
└── types.py          # 数据类

demo/
└── demo.py           # TUI 对话演示

tests/
└── test_all.py       # 60 个测试用例
```

## 许可证

本项目使用 **非商业用途许可**。详情见 [LICENSE](LICENSE)。
商业使用请联系 aa20170612@outlook.com。

## 贡献

欢迎任何形式的贡献：

- 提交 [Issue](https://github.com/your-repo/issues) 报告 bug 或建议
- 提交 PR 改进代码或文档
- 优化参数配置适配不同嵌入模型
- 开发 GPU 版本加速嵌入计算
