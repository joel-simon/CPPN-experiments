import numpy as np
import itertools, random
from numpy.random import randint
from classification import balanced_accuracy
from patterns import pretty_print
import copy

class Individual(object):
    def __init__(self, shape):
        self.shape = shape
        self.genome = randint(0, 2, (shape))
        self.fitness = None

    def mutate(self, mutation_rate):
        num_mutations = int(np.random.normal(mutation_rate*self.genome.size, 1))

        for m in range(num_mutations):
            i = randint(self.shape[0])
            j = randint(self.shape[1])
            self.genome[i, j] = not self.genome[i, j]

    def crossover(self, other):
        assert self.fitness >= other.fitness
        m, n = self.genome.shape
        i0, j0 = randint(0, m), randint(0, n)
        h, w   = randint(0, m), randint(0, n)

        child = copy.deepcopy(self)

        for i in range(i0, i0 + h):
            for j in range(j0, j0 + w):
                i_ = i % m
                j_ = j % n
                child.genome[i_, j_] = other.genome[i_, j_]

        return child

class IndividualNoWrap(Individual):
    def crossover(self, other):
        m, n = self.genome.shape
        i0, j0 = randint(0, m), randint(0, n)
        i1, j1 = randint(i0, m), randint(j0, n)
        child = copy.deepcopy(self)
        child.genome[i0:i1, j0:j1] = other.genome[i0:i1, j0:j1]
        return child

class IndividualRandomCross(Individual):
    def crossover(self, other):
        child = copy.deepcopy(self)

        m, n   = self.shape
        h, w   = randint(0, m), randint(0, n) # Size of crossover block
        ic, jc = randint(0, m), randint(0, n) # Child index offset
        io, jo = randint(0, m), randint(0, n) # Other index offset

        for i in range(h):
            for j in range(w):
                v = other.genome[(i+io)%m, (j+jo)%m]
                child.genome[(i+ic)%m, (j+jc)%m] = v

        return child

class IndividualTwoPointCross(Individual):
    def crossover(self, other):
        child = copy.deepcopy(self)

        genome = child.genome.flatten()
        other = other.genome.flatten()

        i, j = randint(0, genome.size), randint(0, genome.size)

        if j < i:
            length = genome.size - j + i
        else:
            length = j - i

        i = randint(0, genome.size)
        inds = list(range(i, i+length))

        np.put(genome, inds, np.take(other, inds, mode='wrap'), mode='wrap')
        child.genome = genome.reshape(self.shape)

        return child

def GA(pop_size, generations, elitism, target, Individual, mutation_rate, view=False):
    shape = target.shape

    pop = [Individual(shape) for i in range(pop_size)]

    top = []

    for i in range(generations):
        for p in pop:
            p.fitness = balanced_accuracy(target, p.genome)

        children = sorted(pop, key=lambda p: p.fitness, reverse=True)[:elitism]

        for _ in range(pop_size - elitism):
            a, b = random.sample(pop, 2)

            if a.fitness > b.fitness:
                child = a.crossover(b)
            else:
                child = b.crossover(a)

            child.mutate(mutation_rate)
            children.append(child)

        pop = children

        top.append(pop[0])

        if view:
            print(pop[-1].fitness)
            pretty_print(pop[-1].genome)

        if pop[-1].fitness == 1:
            return top

    return top

def main():
    ind1 = IndividualTwoPointCross((8,8))
    ind2 = IndividualTwoPointCross((8,8))
    pretty_print(ind1.genome)
    pretty_print(ind2.genome)
    pretty_print(ind1.crossover(ind2).genome)
    # pop_size = 100
    # generations = 100
    # elitism = 2
    # target = patterns[0]
    # pop = GA(pop_size, generations, elitism, target, Individual, 1/(16*16.), view=True)
    # print(pop[-1].fitness)

if __name__ == "__main__":
    main()
