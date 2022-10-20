

from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search import BayesianOptimizationSearch, PESearch

X, y = cars.load()


automl_bayesian = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=1,
    search_algorithm=BayesianOptimizationSearch
)

automl_pge = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=1,
    search_algorithm=PESearch
)

automl_bayesian.fit(X, y)
automl_pge.fit(X, y)


#print(automl_bayesian.best_pipeline_)
print(automl_bayesian.best_score_)

#print(automl_pge.best_pipeline_)
print(automl_pge.best_score_)