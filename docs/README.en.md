# Knot

**Experimental** — A dynamic memory simulation system using pure graph structures (retrieval + consolidation).

Knot simulates the brain's memory reading and consolidation using a directed weighted graph, without any neural network gradient training.

> **⚠️ Experimental Notice**
>
> This project is in early experimental stage. APIs and internals may change as we learn from real usage.
> All contributions are welcome: tweak the code, tune parameters, submit PRs, open issues.
> GPU-accelerated versions are especially welcome to speed up embedding computation (the author is poor and only has CPU 🙃).

## Quick Start

```python
from knot import HippocampusMemorySystem

system = HippocampusMemorySystem()
system.consolidate("(Python, is, programming language)", query="init")
result = system.retrieve("Python")
```

## Documentation

| Doc | Content |
|-----|---------|
| [Technical Spec](docs/technical.md) | Architecture, core concepts, formulas |
| [Usage Guide](docs/usage.md) | Installation, config, examples |
| [API Reference](docs/api-reference.md) | All classes and methods |
| [Tuning Guide](docs/tuning.md) | Hyperparameters explained |
| [Demo App](docs/demo.md) | TUI conversation demo |
| [中文文档](README.md) | Chinese documentation |

## Project Structure

```
knot/
├── graph.py          # Graph structure (node/edge CRUD)
├── retrieval.py      # Retrieval phase (RWR, seed selection)
├── consolidation.py  # Consolidation phase (edge decision tree, bias drift)
├── knot/memory_system.py  # Main orchestrator
├── embedding.py      # Embedding interface (Qwen3 / Dummy)
├── config.py         # Hyperparameters
└── types.py          # Data classes

demo/
└── demo.py           # TUI conversation demo

tests/
└── test_all.py       # 60 test cases
```

## License

This project is licensed under a **non-commercial license**. See [LICENSE](../LICENSE) for details.
For commercial use, contact aa20170612@outlook.com.

## Contributing

Any form of contribution is welcome:

- Submit [Issues](https://github.com/your-repo/issues) for bugs or suggestions
- Submit PRs for code or documentation improvements
- Tune parameters for different embedding models
- Develop GPU-accelerated versions for faster embedding computation
