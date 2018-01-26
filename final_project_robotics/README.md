# DQL with NoisyNet

This folder contains an extension of the [OpenAI baselines Deep Q-Learning (d8cce2309f)](https://github.com/openai/baselines/tree/d8cce2309f3765bf55c46e4ffe4722406f412275) implementing [NoisyNet](https://arxiv.org/pdf/1706.10295.pdf) as an replacement of $\epsilon$ -greedy exploration policy.

## Installation

We recommend a Linux system with `python3` installed.
Use these commands in your command line interface (*shell*).

```
# Download the repository and open this folder. 
git clone https://github.com/dekuenstle/dl_lab_2017.git
cd dl_lab_2017/final

# Install the (extended) baselines package.
cd baselines
pip3 install -e .
cd ..
```

## Atari

Train a DQL agent on Atari. This will take hours even on a GPU. 

```
# Create save folder
mkdir models
# Start training
python3 -m baselines.deepq.experiments.atari.train --env=Breakout --save-dir=models
```

See `python3 -m baselines.deepq.experiments.atari.train --help` for more parameters.

After training, you can enjoy the trained agent playing the game and store a video of it.

```
# Get latest model
LATEST_MODEL=$(ls -d models/model-* | sort -V | tail -n1)
echo "${LATEST_MODEL}"

# Enjoy agent and record.
python3 -m baselines.deepq.experiments.atari.enjoy --env=Breakout --model-dir=${LATEST_MODEL} --video=breakout.mp4
```

## Development

The `./baselines` folder contain only modified parts of the OpenAI baselines repository.
See `./baselines/LICENCE` for their terms of use.

Most modifications took place in the DQL implementation `./baselines/baselines/deepq`:

- `simple.py` Q-Learning algorithm and interface
- `models.py` Deep-Q-model implementation
- `build_graph.py` Tensor graph combining Q-model with loss etc. 
