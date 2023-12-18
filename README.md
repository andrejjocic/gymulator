# gymulator

This repository explores the problem of *gym layout optimisation*.
A common occurence in gyms at peak hours is over-crowding of popular equipment and lots of wandering around searching for a free machine.
Through simulating gym-goer behavior, we aim to identify the most effective arrangement of exercise equipment to improve the flow, accessibility, and overall customer satisfaction.
Our goal is to offer practical insights for gym owners, managers, and designers seeking to optimize their facility layout for the benefit of their clients and business success.


## Project structure
We are modelling gym dynamics with Python's *Mesa* library for agent-based model simulation and visualization. 

The source files in the `/model/` directory are:

- `gym_model.py`: This file contains a subclass of `mesa.Model`, which represents the gym environment. It includes the layout of the gym and the equipment available.

- `gym_agent.py`: This file contains a subclass of `mesa.Agent`, which represents a gym-goer. It includes the behavior of the gym-goer, such as choosing equipment and exercising.

- `visualisation.py`: This file contains functions for visualizing a gym environment.

## Milestones
- [**20th of November, 2023**: formulated model of gym-goer behaviour](https://github.com/andrejjocic/gymulator/milestone/1)
- [**18th of December, 2023**: simulations done](https://github.com/andrejjocic/gymulator/milestone/2)
- [**8th January, 2024**: simulation analysis done](https://github.com/andrejjocic/gymulator/milestone/3)

## Authors
- [Andrej Jočić](https://github.com/andrejjocic)
- [Matic Stare](https://github.com/maticstare)
- [Martin Starič](https://github.com/SpongeMartin)

