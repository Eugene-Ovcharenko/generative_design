
import numpy as np
import pygad

from tools.utils import calculate_design_score
from tools.model import Regressor



def scoring_function(params, idx) -> float:
#[HGT: float, DIA: float, ANG: float, CVT: float, THK: float, ELM: float, UTS: float, idx]
    lumen_model_path = 'models/LMN_compete_rmse_Raw_Power_Split_1117_0306'
    stress_model_path = 'models/Smax_compete_rmse_MinMax_Power_Split_1056_0306'
    model_lumen = Regressor(model_path=lumen_model_path)
    model_stress = Regressor(model_path=stress_model_path)

    _data_point = np.array(
        [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5]
        ]
    )
    data_point = _data_point.reshape(1, -1)

    lumen = float(model_lumen(data_point))
    lumen = 0 if lumen < 0 else lumen

    stress = float(model_stress(data_point))
    stress = 0 if stress < 0 else stress

    score = calculate_design_score(
        lumen_abs=lumen,
        stress_abs=stress,
        uts=22.5,
    )
    return score


if __name__ == '__main__':

    BOUNDS = [{'low': 15,   'high': 25},    # valve height
              {'low': 25,   'high': 30},    # valve diameter
              {'low': -30,  'high': 30},    # free edge angle
              {'low': 0,    'high': 1},     # leaflet curvature
              [0.3],                        # leaflet thickness
              [1.9]]                        # material ELM

    INIT = [[15, 25, -30, 0, 0.3, 1.9], # initial population for test only
            [18, 26, 0, 0.4, 0.3, 1.9],
            [21, 27, 15, 0.8, 0.3, 1.9],
            [24, 28, 30, 1, 0.3, 1.9]]

    ga_instance = pygad.GA(fitness_func=scoring_function, # scoring_function tune
                           num_generations=1,             # iterations number
                           num_genes=6,                   # genes number
                           gene_type=float,               # Controls the gene type
                           gene_space=BOUNDS,             # Parameters range
                           sol_per_pop=4,                 # chromosomes number
                           num_parents_mating=2,          # num
                           # initial_population=INIT,     # initial population for test only
                           parent_selection_type='rank',  # parents rank selection method
                           crossover_type='uniform',      # uniform crossover
                           crossover_probability=0.5,     # crossover probability 0...1
                           mutation_type=None,            #
                           # mutation_type="adaptive",    #
                           # mutation_probability=[0, 0], #
                           save_solutions=True)           #



    # popul=ga_instance.initialize_population(low=0, high=1, allow_duplicate_genes=True, gene_type=float, mutation_by_replacement=False)
    # print(popul)
    ga_instance.run()
    print("Initial population: \n {init_pop}\n".format(init_pop=ga_instance.initial_population))

    print(ga_instance.solutions)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    if ga_instance.best_solution_generation != -1:
        print("Best fitness value reached after {best_solution_generation} generations.".format(
            best_solution_generation=ga_instance.best_solution_generation))

    ga_instance.cal_pop_fitness()
    ga_instance.plot_fitness()                          # Shows how the fitness evolves by generation.
    # ga_instance.plot_genes(plot_type="scatter")         # Shows how the gene value changes for each generation.
    # ga_instance.plot_new_solution_rate()                # Shows the number of new solutions explored in each

    ga_instance.solutions
    prediction = np.sum(scoring_function(solution, 0))
    print(prediction)

