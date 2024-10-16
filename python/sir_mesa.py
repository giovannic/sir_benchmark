# Write an SIR model using the mesa framework

# Import libraries
import sys
import enum
import random
import csv
import pandas as pd

# Import classes
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

# Read in command line arguments
N = int(sys.argv[1])
parameters = sys.argv[2]
output_file = sys.argv[3]

class State(enum.IntEnum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class InfectionModel(Model):
    """ A model for disease spread """

    def __init__(self, N, R0, I0, gamma):
        self.N = N
        self.gamma = gamma
        self.beta = R0 * gamma
        self.I0 = I0
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(          
            agent_reporters={
                "S": lambda model: sum(1 for a in model.schedule.agents if a.state == State.SUSCEPTIBLE),
                "I": lambda model: sum(1 for a in model.schedule.agents if a.state == State.INFECTED),
                "R": lambda model: sum(1 for a in model.schedule.agents if a.state == State.REMOVED),
            }
        )
        self.agents = []

        # Create agents
        infected = random.choices(list(range(self.N)), k=self.I0)
        for i in range(self.N):
            a = Person(i, self)
            if i in infected:
                a.state = State.INFECTED
            self.agents.append(a)

        self.foi = self.calculate_foi()

    def calculate_foi(self):
        """ Calculate force of infection """
        num_infected = sum([
            1 for a in self.agents if a.state == State.INFECTED
        ])
        return self.beta * num_infected / self.N

    def set_foi(self):
        """ Set force of infection """
        self.foi = self.calculate_foi()

    def step(self):
        self.set_foi()
        self.datacollector.collect(self)

# Create agent class
class Person(Agent):

    def __init__(self, unique_id, model: InfectionModel):
        super().__init__(unique_id, model)
        self.state = State.SUSCEPTIBLE
        self.model = model

    def infect(self):
        """ Infect agent """
        if self.state == State.SUSCEPTIBLE:
            if random.random() < self.model.foi:
                self.state = State.INFECTED

    def recover(self):
        """ Recover agent """
        if self.state == State.INFECTED:
            if random.random() < self.model.gamma:
                self.state = State.REMOVED

    def step(self):
        """ Step method for agent """
        self.infect()
        self.recover()

if __name__ == '__main__':

    # Read in parameters from file
    with open(parameters, 'r') as f:
        reader = csv.reader(f)
        outputs = []
        for row in reader:
            # Run model
            model = InfectionModel(
                N=N,
                R0=int(N * float(row[0])),
                I0=int(N * float(row[1])),
                gamma=float(row[2])
            )

            dt = 0.1
            tmax = 100
            for j in range(int(tmax / dt)):
                model.step()

            # Save output
            run_output = model.datacollector.get_agent_vars_dataframe()
            outputs.append(run_output)

    # Save output
    all_outputs = pd.concat(outputs)
    all_outputs.to_csv(output_file, index=False)