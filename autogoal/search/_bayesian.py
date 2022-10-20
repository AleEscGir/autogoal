
from ._base import SearchAlgorithm
from typing import Mapping, Optional, Dict, List, Sequence
from random import random
from autogoal.sampling._bayesianModelSampler import BayesianModelSampler, clubster_by_epsilon, Bernoulli
from autogoal.sampling import best_indices
import random


class BayesianOptimizationSearch(SearchAlgorithm):
    def __init__(self, 
                *args,
                alpha: float = 0,
                alpha_increment : float = 0.01,
                alpha_reiniciation : float = 0,
                selection : float = 0.2,
                epsilon : float = 0.1,
                random_state: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs) 
        
        self._alpha = alpha
        self._alpha_increment = alpha_increment
        self._alpha_reiniciation = alpha_reiniciation
        self._selection = selection
        self._epsilon = epsilon
        self._random_states = random.Random(random_state)
        self._model: Dict = {}

    def _start_generation(self):
        self._samplers = []

    def _build_sampler(self):

        if len(self._samplers) == 0:
            sampler = BayesianModelSampler(alpha=self._alpha, exploration=True)
        else:
            self._model = self._samplers[len(self._samplers) - 1]._model
            sampler = BayesianModelSampler(self._model, self._alpha, exploration=not(Bernoulli(self._alpha)))

        self._alpha += self._alpha_increment

        if self._alpha >= 1:
            self._alpha = self._alpha_reiniciation

        self._samplers.append(sampler)
        return sampler


    def _finish_generation(self, fns):

        # Vamos a buscar aquel alfa que tenga la mayor
        # cantidad de valores en una vecindad de epsilon

        indices = best_indices(fns, k=int(self._selection * len(fns)), maximize=self._maximize)

        samplers_values = [self._samplers[i]._alpha for i in indices]

        clubsters = clubster_by_epsilon(samplers_values, self._epsilon)
        
        max_value = 0
        pos = 0

        for i in range(len(clubsters)):
            if clubsters[i] > max_value:
                max_value = clubsters[i]
                pos = i
        
        self._alpha = samplers_values[pos]

        
    
