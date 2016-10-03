# Human-Level Control through Deep Reinforcement Learning

Tensorflow implementation of [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf).

This implementation contains:

1. Deep Q-network and Q-learning
2. Experience replay memory
    - to reduce the correlations between consecutive updates
3. Network for Q-learning targets are fixed for intervals
    - to reduce the correlations between target and predicted Q-values

Refer to [devsisters/DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)


## Usage

- Install [Docker](https://docs.docker.com/engine/installation/linux/ubuntulinux/)

- Create container

    $ docker run -p 8888:8888 -it --name=dqntest --volume=/home/nsilva/gitrepos/projectappia/dqn:/shared projectappia/dqn:latest-cpu

- Start already created container

    $ docker start -ai dqntest

- Access container bash (new terminal)
    
    $ docker exec -it dqntest1 bash

- Connect a fake display (inside container bash)

    $ xvfb-run -s "-screen 0 1400x900x24" bash

- Run a model to test (inside container bash)

    $ python main.py --is_train=False --display=True --use_gpu=False

More check 'useful_commands'

## Results

Result of training for 24 hours using GTX 980 ti.

![best](assets/best.gif)


## References

- [devsisters/DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow)
- [simple_dqn](https://github.com/tambetm/simple_dqn.git)
- [Code for Human-level control through deep reinforcement learning](https://sites.google.com/a/deepmind.com/dqn/)


## License

MIT License.
