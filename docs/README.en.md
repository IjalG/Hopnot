# Hopnot

**Experimental** — A dynamic memory simulation system using pure graph structures (retrieval + consolidation).

Hopnot simulates the brain's memory reading and consolidation using a directed weighted graph, without any neural network gradient training.

> **⚠️ Experimental Notice**
>
> This project is in early experimental stage. APIs and internals may change as we learn from real usage.
> All contributions are welcome: tweak the code, tune parameters, submit PRs, open issues.
> GPU-accelerated versions are especially welcome to speed up embedding computation (the author is poor and only has CPU 🙃).

## Quick Start

```python
from hopnot import HippocampusMemorySystem

system = HippocampusMemorySystem()
system.consolidate("(Python, is, programming language)", query="init")
result = system.retrieve("Python")
```

## Documentation

| Doc | Content |
|-----|---------|
| [Technical Spec](technical.en.md) | Architecture, core concepts, formulas |
| [Usage Guide](usage.en.md) | Installation, config, examples |
| [API Reference](api-reference.en.md) | All classes and methods |
| [Tuning Guide](tuning.en.md) | Hyperparameters explained |
| [Demo App](demo.en.md) | TUI conversation demo |
| [中文文档](../README.md) | Chinese documentation |

## Project Structure

```
hopnot/
├── graph.py          # Graph structure (node/edge CRUD)
├── retrieval.py      # Retrieval phase (RWR, seed selection)
├── consolidation.py  # Consolidation phase (edge decision tree, bias drift)
├── memory_system.py  # Main orchestrator
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

- Submit [Issues](https://github.com/IjalG/Hopnot/issues) for bugs or suggestions
- Submit PRs for code or documentation improvements
- Tune parameters for different embedding models
- Develop GPU-accelerated versions for faster embedding computation
