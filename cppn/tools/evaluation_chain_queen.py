import hashlib
from lxml import etree
import subprocess as sub
import numpy as np


def evaluate_population(pop):

    num_evaluated_this_gen = 0

    for n, ind in enumerate(pop):

        # don't evaluate if invalid
        if not ind.phenotype.is_valid():
            for rank, goal in pop.objective_dict.items():
                if goal["name"] != "age":
                    setattr(ind, goal["name"], goal["worst_value"])

            print "Skipping invalid individual"

        # otherwise create a robot file
        else:
            num_evaluated_this_gen += 1
            pop.total_evaluations += 1

            (x, y, z) = ind.genotype.orig_size_xyz

            for name, details in ind.genotype.to_phenotype_mapping.items():
                if name == "Data":
                    body = details["state"]

            # md5 so we don't eval the same robot more than once
            m = hashlib.md5()
            m.update(str(body))
            ind.md5 = m.hexdigest()

            # don't evaluate if identical phenotype has already been evaluated
            if ind.md5 in pop.already_evaluated:

                for rank, goal in pop.objective_dict.items():
                    if goal["tag"] is not None:
                        setattr(ind, goal["name"], pop.already_evaluated[ind.md5][rank])

                print "Age {0} individual already evaluated: cached fitness is {1}".format(ind.age, ind.fitness)

            else:
                # save the robot files in a data file
                with open('data'+str(seed)+'/bot_{:04d}'.format(ind.id), 'wb') as f:
                    f.write(str(body))
  
    # evaluate all the robots in the data directory
    sub.call("./ChainQueen -i data{0} -o output{1}.xml".format(seed, seed), shell=True)
    root = etree.parse("output{}.xml".format(seed)).getroot()
    
    # get the results from ChainQueen
    for ind in pop:

        if ind.phenotype.is_valid() and ind.md5 not in pop.already_evaluated:

            # assign the fitness to the individual
            ind.fitness = float(root.findall("detail/bot_{:04d}/fitness_score".format(ind.id))[0].text)

            print "Assigning ind {0} fitness {1}".format(ind.id, ind.fitness)

            pop.already_evaluated[ind.md5] = [getattr(ind, details["name"])
                                              for rank, details in
                                              pop.objective_dict.items()]


