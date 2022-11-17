
import math
import os

from autogoal.datasets import abalone, cars, cifar10, dorothea, dummy, german_credit, gisette, haha, meddocan, movie_reviews, shuttle, wine_quality, yeast
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import BayesianOptimizationSearch, PESearch, Logger

X, y = cars.load()

#Logger para imprimir en consola todos los fitness
class Experimentation_Print_Logger(Logger):

    def __init__(self):
        self.fitness_count = 0
        self.errors_count = 0
        self.generation_best_fitness = 0

    def start_generation(self, generations, best_fn):
        self.fitness_count += 1
        self.errors_count = 0
        self.generation_best_fitness = 0
        #print("\n")
        #print("Generation", self.fitness_count)
        

    def eval_solution(self, solution, fitness):
        if fitness in [-math.inf, math.inf, 0]:
            self.errors_count += 1
        else: 
            if self.generation_best_fitness < fitness:
                self.generation_best_fitness = fitness
            #print("Fitness:", fitness)

    def finish_generation(self, fns):
        #print("Number of Errors:", self.errors_count)
        print("Generation", self.fitness_count, "Best Fitness", self.generation_best_fitness)
        self.generation_best_fitness = 0

    def end(self, best, best_fn):
        #print("Number of Errors:", self.errors_count)
        print("Generation", self.fitness_count, "Best Fitness", self.generation_best_fitness)
        print("\n")
        print("Global Best Fitness", best_fn)


    def error(self, e: Exception, solution):
        pass

def Test_Funtion(a : float, b : float, c : float, d : float, e : bool, f : bool, g : bool, h: bool):
    sum = 0

    if e:
        sum += a
    if f:
        sum -= b
    if g:
        sum *= c
    if h and d != 0:
        sum /= d


automl_bayesian = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=100,
    search_algorithm=BayesianOptimizationSearch,
    alpha = 0,
    alpha_increment = 0.1,
    alpha_reiniciation = 0
)

automl_pge = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=100,
    search_algorithm=PESearch
)

automl_bayesian.fit(X, y, logger = Experimentation_Print_Logger())
#automl_pge.fit(X, y, logger = Experimentation_Print_Logger())


#print(automl_bayesian.best_pipeline_)
#print(automl_bayesian.best_score_)

#print(automl_pge.best_pipeline_)
#print(automl_pge.best_score_)