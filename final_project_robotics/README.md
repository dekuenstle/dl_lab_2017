# DQN-Baselines with NoisyNet

This folder contains an extension of the [OpenAI baselines Deep Q-Learning](https://github.com/openai/baselines/) implementing [NoisyNet](https://arxiv.org/pdf/1706.10295.pdf) as an replacement of $\epsilon$ -greedy exploration policy.

## Installation

We recommend a Linux system with `python3` installed.
Use these commands in your command line interface (*shell*).

```
# Download the repository and open this folder. 
git clone https://github.com/dekuenstle/dl_lab_2017.git
cd dl_lab_2017/final_project_robotics

# Install the (extended) baselines package.
cd baselines
pip3 install -e .
cd ..
```

## Experiments

In the `./experiments/` folder we defined some experiments to compare different exploration technology.
$\epsilon$-*greedy* forms the baseline, more interestingly we run the similar but different agent noise techniques:
*parameter noise* by *OpenAI* and *noisy network* by *Google DeepMind*.

Simply run the files in the folder to reproduction, or simply have a look to the poster `./poster.pdf`.

## Development

The `./baselines` folder contain only modified parts of the OpenAI baselines repository.
See `./baselines/LICENCE` for their terms of use.

### Important files
In `./baselines/baselines/deepq`:

- `simple.py` Q-Learning algorithm and interface
- `models.py` Deep-Q-model as multilayer perceptron
- `build_graph.py` Tensor graph combining Q-model with exploration policy etc. 

In `./baselines/baselines/common`:

- `tf_util.py` Utilities (also noise layer definition)
