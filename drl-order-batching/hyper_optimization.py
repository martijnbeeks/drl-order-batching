'''
Code for the hyper optimization framework Hyperspace that optimizes parameters of a reward function.
'''
from hyperspace.hyperdrive import hyperdrive
from hyper_objective import ObjectiveFunction
from hyperspace.kepler.data_utils import load_results


def main():
    hparams = [(0.1, 1.5),      # weight tardy orders
               (0.1, 1.5)]      # weight picking time

    objective_function = ObjectiveFunction()

    hyperdrive(objective=objective_function.objective,
               hyperparameters=hparams,
               results_path='optimization_results/scenario_2',
               checkpoints_path='optimization_results/scenario_2',
               model="GP",
               n_iterations=50,
               verbose=True)

    path = './optimization_results/scenario_3'
    results = load_results(path, sort=True)
    print('Hyperparameters of our best model:\n {}'.format(results[0].x))


if __name__ == '__main__':
    main()
