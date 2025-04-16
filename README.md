# 🎮 JAXtari: JAX-Based Object-Centric Atari Environments

Quentin Delfosse, Daniel Kirn, Dominik Mandok, Paul Seitz, Lars Teubner, Sebastian Wette  
[Machine Learning Lab – TU Darmstadt](https://www.ml.informatik.tu-darmstadt.de/)

> A GPU-accelerated, object-centric Atari environment suite built with JAX for fast, scalable reinforcement learning research.

---

**JAXtari** introduces a GPU-accelerated, object-centric Atari environment framework powered by [JAX](https://github.com/google/jax). Inspired by [OCAtari](https://github.com/k4ntz/OC_Atari), this framework enables up to **16,000x faster training speeds** through just-in-time (JIT) compilation, vectorization, and massive parallelization on GPU.

<!-- --- -->

## Features
- Object-centric extraction of Atari game states.
- JAX-based vectorized execution with GPU support.
- Compatible API with ALE to ease integration.
- Benchmarking tools.

<!-- [**📘 JAXtari Documentation**] -->

## Getting Started

<!-- ### Prerequisites -->
### Install
```bash
python3 -m venv .venv
source .venv/Scripts/activate

python3 -m pip install -U pip
pip3 install "gymnasium[atari, accept-rom-license]"
pip3 install -e .
```

## Usage

Using an environment:
```python
from jaxtari import JAXtari

env = JAXtari("pong")
state = env.get_init_state()
state = env.step_state_only(state, action=0)
```


Running a game manually:
```bash
python3 -m jaxtari.games.jax_seaquest
```

---

## Supported Games

| Game      | Supported |
|-----------|-----------|
| Seaquest  | ✅        |
| Pong      | ✅        |

> More games can be added via the uniform wrapper system.

---

## Contributing

Contributions are welcome!

1. Fork this repository  
2. Create your feature branch: `git checkout -b feature/my-feature`  
3. Commit your changes: `git commit -m 'Add some feature'`  
4. Push to the branch: `git push origin feature/my-feature`  
5. Open a pull request  

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---
