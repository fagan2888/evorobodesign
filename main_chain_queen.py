import random
import numpy as np
import sys
from time import time
import cPickle
import subprocess as sub
from glob import glob

from cppn.networks import CPPN
from cppn.softbot import Genotype, Phenotype, Population
from cppn.tools.algorithms import Optimizer
from cppn.tools.utilities import natural_sort, make_one_shape_only
from cppn.objectives import ObjectiveDict
from cppn.tools.evaluation_chain_queen import evaluate_population
from cppn.tools.mutation import create_new_children_through_mutation
from cppn.tools.selection import pareto_selection


SEED = int(sys.argv[1])
random.seed(SEED)
np.random.seed(SEED)

GENS = 1001
POPSIZE = 50

IND_SIZE = (8, 8, 7)

CHECKPOINT_EVERY = 100  # gens
MAX_TIME = 47  # [hours] evolution does not stop; after MAX_TIME, checkpointing occurs at every generation.

DIRECTORY = "."


def one_muscle(output_state):
    return make_one_shape_only(output_state) * 1


class MyGenotype(Genotype):

    def __init__(self):

        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["Data"]))
        self.to_phenotype_mapping.add_map(name="Data", tag="<Data>", func=one_muscle, output_type=int)


class MyPhenotype(Phenotype):

    def is_valid(self):
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                print "INVALID: Nans in phenotype."
                return False

            if name == "Data":
                state = details["state"]

                # make sure there is some material to simulate
                if np.sum(state) < 5:
                    print "INVALID: Empty sim."
                    return False

        return True


# The objectives to be optimized
my_objective_dict = ObjectiveDict()
# maximize fitness:
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<fitness_score>")
# minimize the age of solutions: promotes diversity:
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

# quick test here to make sure evaluation is working properly:
# evaluate_population(my_pop)
# print [ind.fitness for ind in my_pop]

if len(glob("pickledPops{}/Gen_*.pickle".format(SEED))) == 0:
    # initialize a population of SoftBots
    my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE, material_wide_phase_offset=True, seed=SEED)
    # start evolution
    my_optimization = Optimizer(my_pop, pareto_selection, create_new_children_through_mutation, evaluate_population)

else:  # continue from checkpoint if there is one saved
    successful_restart = False
    pickle_idx = 0
    while not successful_restart:
        try:
            pickled_pops = glob("pickledPops{}/*".format(SEED))
            last_gen = natural_sort(pickled_pops, reverse=True)[pickle_idx]
            with open(last_gen, 'rb') as handle:
                [optimizer, random_state, numpy_random_state] = cPickle.load(handle)
            successful_restart = True

            my_pop = optimizer.pop
            my_optimization = optimizer
            my_optimization.continued_from_checkpoint = True
            my_optimization.start_time = time()

            random.setstate(random_state)
            np.random.set_state(numpy_random_state)

            print "Starting from pickled checkpoint: generation {}".format(my_pop.gen)

        except EOFError:
            # something went wrong writing the checkpoint : use previous checkpoint and redo last generation
            sub.call("touch IO_ERROR_$(date +%F_%R)", shell=True)
            pickle_idx += 1
            pass


my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=GENS, checkpoint_every=CHECKPOINT_EVERY, directory=DIRECTORY)


