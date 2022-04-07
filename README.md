<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">Deep reinforcement learning-based solution for a multi-objective online order batching problem</h3>

  <p align="center">
    Martijn Beeks, Reza Refaei Afshar, Yingqian Zhang, Remco Dijkman, Claudy van Dorst, Stijn de Looijer
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains code for the presented DRL approach in the paper "Deep reinforcement learning-based solution for a multi-objective online order batching problem" by 



<!-- GETTING STARTED -->
## Getting Started

This is an example on how to setup this project locally on python 3.7.10.


1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install pip packages
   ```sh
   python -m pip install -r requirements.txt
   ```
3. Install Hyperspace
    ```sh
   cd drl-order-batching
   git clone https://github.com/yngtdd/hyperspace
   cd hyperspace
   python -m pip install .
   ```


<!-- USAGE EXAMPLES -->
## Usage

Similar to the paper, this repository consists out of several approaches.
1. Simulation model to test DRL approach
2. DRL approach
3. Hyper-parameter optimization of DRL approach

####1. Simulation model
This simulation model models processes within a warehousing concept. The script drl_env.py defines a gym enviroment of the 
simulation model and enables it to be controlled by a variety of sequential decision-making algorithms. Changing
the scenario for the simulation can also be done here.
The environment is in a certain state when an agent predicts an action that moves the environment to a new state. 
This simulation model can be interchanged with different environments. To control the simulation model to test for 
example a heuristic, simulation_control.py can be used. Within this script, a scenario, heuristic and order data
can be defined and a simulation run presents the performance metrics of this scenario.
   ```sh
   python -m simulation_control.py
   ```

####2. DRL approach
A DRL approach for the warehousing problem has been trained with a stable-baselines library. This library requires a gym
environment for the models. The script drl_train.py defines a training job by setting the number of steps to take,
hyper-parameters of the learning model and etc. The scenario for this training job can be defined in drl_env.py where 
a yaml file with a scenario is loaded. When the training job is done, the model is saved at a predefined location.
To test a trained DRL approach, use drl_test.py and define the location of a trained model. This script will output the 
performance of a trained model in the console.
   ```sh
   python -m drl_train.py
   python -m drl_test.py
   ```

####3. Hyper-parameter optimization of DRL approach
As presented in the paper, the reward function has been parametrized and optimized with the hyper-optimization framework
Hyperspace. This framework performs a Bayesian Optimization approach to the solution space and provides us with a set of
parameters or weights. In order to use this framework with the simulation environment of the warehousing concept,
two scipts have been provided. The script hyper_objective.py defines an objective function for the Bayesian Optimization 
framework to optimize. This objective function inputs the two weight parameters of the reward function and simulates
an entire simulation with these parameters. The objective function will subsequently output the performance metrics of 
simulation run. In this script, a certain pre-trained model can be selected. This optimization approach can be started 
with hyper_optimization.py. Within this script, the parameter ranges can be set, and an output location of the 
optimization results. When the optimization framework is done, the two optimal parameters are printed in the console.
   ```sh
   python -m hyper_optimization.py
   ```

These parameters can be used as input when training a DRL approach using drl_train.py by 
changing params: WAREHOUSE(params=(1, 1)).