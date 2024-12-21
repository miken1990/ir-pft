# Previous Knowledge Utilization In Online Anytime Belief Space Planning

This repository contains the code accompanying the paper:  
**[Previous Knowledge Utilization In Online Anytime Belief Space Planning](https://arxiv.org/abs/2412.13128)**  

If you use this code in your research, please cite our work:  

```bibtex
@article{Novitsky2024arxiv,
  title={Previous Knowledge Utilization In Online Anytime Belief Space Planning},
  author={Novitsky, M. and Barenboim, M. and Indelman, V.},
  year={2024},
  eprint={2412.13128},
  archivePrefix={arXiv},
}
```

## Description
This repository contains an implementation of the IR-PFT (Incremental Reuse Particle Filter Tree) algorithm presented in the paper.
The IR-PFT algorithm leverages prior knowledge to enhance efficiency in belief space planning.

## Prerequisites
The code is written in Julia. To run the code, follow the following steps:
- Install julia>1.7.0
- Install environment
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Usage
To execute the main script for replicating the experiments described in the paper, run:
```bash
julia run_script.jl
```