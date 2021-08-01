# Ape-X DQN 

## Description

Recreating the Ape-X Deep Q-Network, a distributed architecture for deep reinforcement learning proposed by DeepMind, with tensorflow and applying on the OpenAI gym environments

## Getting Started

### Dependencies

* Developed on macOS Mojave with Python 3.8.8

### Installing

* Clone or download the repository
* Install required modules with
```
pip install -r requirements.txt
```
* Change the build_model function in DQNagent.py file to fit your need

### Executing program

* Run example on CartPole-v0 environment
```
python example.py
```

## Help

Email us for more detail

## Authors

Andrew Yu - [Github](https://github.com/yuyenchu) - andrew7011616@gmail.com

Haley Lin (coeditor) - [Github](https://github.com/HaleyLin2006) - haleylin2006@gmail.com

## Version History

* 0.1-beta
    * Multi-thread version complete
* 0.1-alpha
    * Prototype Complete (partial functions incomplete)
* 0.0-alpha
    * In Development (incomplete)

## License

This project is licensed under the GNU License - see the LICENSE.md file for details

## Acknowledgments

Inspiration and snippet from
* [Distributed Prioritized Experience Replay](https://arxiv.org/pdf/1803.00933v1.pdf)
* [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* [Multi-step Bootstrapping](https://www.cs.ubc.ca/labs/lci/mlrg/slides/Multi-step_Bootstrapping.pdf)
* [Implementing Deep Reinforcement Learning Models with Tensorflow + OpenAI Gym](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#deep-q-network)
* [Deep Q-Learning for Atari Breakout](https://keras.io/examples/rl/deep_q_network_breakout/)