"""
Module for simulation of open queueing networks.
"""

import logging
import numpy as np

class Server:
    """
    A server consists of a queue, with arrivals, and communication channels to other servers.
    There does not a dedicated channel class, for the sake of brevity.
    """

    def __init__(self, name, external_arrival_rate, service_rate):
        """
        Initializes an idle server, with a given arrival rate and service rate.
        """

        self.name = name
        self.external_arrival_rate = external_arrival_rate
        self.service_rate = service_rate
        self.queue = []
        self.timestamp = 0
        self.is_busy = False
        self.service_remaining = 0
        self.next_arrival = np.random.exponential(1 / self.external_arrival_rate)

        # initialized later
        self.channels = []
        self.routing_probabilities = []
        self.self_loop_probability = 0
        self.out_probability = 0

    def add_channels(self, servers, routing_probabilities, self_loop_probability):
        """
        Adds links to all servers connected via an outgoing channel.
        """

        self.channels = servers
        self.routing_probabilities = routing_probabilities
        self.self_loop_probability = self_loop_probability
        self.out_probability = 1 - sum(self.routing_probabilities) - self.self_loop_probability
        # print(self.routing_probabilities, self.self_loop_probability, self.out_probability)

    def num_jobs(self):
        """
        Returns the total number of jobs with the server.
        """

        if self.is_busy:
            return len(self.queue) + 1
        return len(self.queue)

    def run(self, time_delta):
        """
        Runs the server for one time step.
        """

        self.timestamp += time_delta
        self.check_external_arrivals(time_delta)
        self.service_current_job(time_delta)
        if not self.is_busy:
            self.draw_job()

    def check_external_arrivals(self, time_delta):
        """
        Checks if the exponential timer for external arrival is up.
        """
        if self.next_arrival > time_delta:
            self.next_arrival -= time_delta
        else:
            logging.debug('%.2f %d: Queued an external arrival', self.timestamp, self.name)
            self.queue.append(0)
            self.is_busy = True
            self.next_arrival = np.random.exponential(1 / self.external_arrival_rate)

    def service_current_job(self, time_delta):
        """
        Checks if the exponential timer for current job is up, routing it as necessary.
        """

        if self.is_busy:
            if self.service_remaining > time_delta:
                self.service_remaining -= time_delta
            else:
                logging.debug('%.2f %d: Serviced a job', self.timestamp, self.name)
                self.service_remaining = 0
                self.is_busy = False
                self.route_job()

    def draw_job(self):
        """
        Draws a job from the queue, assuming the server is idle.
        """

        if self.queue:
            logging.debug('%.2f %d: Started servicing a job', self.timestamp, self.name)
            self.queue.pop(0)
            self.service_remaining = np.random.exponential(1 / self.service_rate)
            self.is_busy = True
        self.is_busy = False

    def route_job(self):
        """
        Routes a job to a server, connected via a channel.
        """
        cdf = np.random.uniform()
        cdf_i = 0
        for i, prob in enumerate(self.routing_probabilities):
            cdf_i += prob
            if cdf <= cdf_i:
                logging.debug('%.2f %d: Routed job to server %d', self.timestamp, self.name, i)
                self.channels[i].queue.append(0)
                return

        cdf_i += self.self_loop_probability
        if cdf <= cdf_i:
            logging.debug('%.2f %d: Routed job to self', self.timestamp, self.name)
            self.queue.append(0)
            self.is_busy = True
            return

        logging.debug('%.2f %d: Job left the system', self.timestamp, self.name)

class Network:
    """
    A network encapsulates all the servers, and provides a framework for synchronous simulation.
    """

    def __init__(self, arrival_rates, service_rates, routing_matrix):
        """
        Initializes all the servers of the network.
        """

        self.validate_input(arrival_rates, service_rates, routing_matrix)

        self.arrival_rates = arrival_rates
        self.service_rates = service_rates
        self.routing_matrix = routing_matrix
        self.servers = []

        for i, _ in enumerate(self.routing_matrix):
            server = Server(i, arrival_rates[i], service_rates[i])
            self.servers.append(server)

        for i, _ in enumerate(self.routing_matrix):
            self.servers[i].add_channels(
                self.servers[:i] + self.servers[i + 1:],
                self.routing_matrix[i][:i] + self.routing_matrix[i][i + 1:],
                self.routing_matrix[i][i]
            )

    def validate_input(self, arrival_rates, service_rates, routing_matrix):
        """
        Checks if rates are positive and routing matrix is a valid transition matrix.
        """
        for i in arrival_rates:
            if i <= 0:
                raise ValueError('Arrival rate should be positive.')
        for i in service_rates:
            if i <= 0:
                raise ValueError('Service rate should be positive.')
        for row in routing_matrix:
            cdf = 0
            for col in row:
                if col < 0 or col > 1:
                    raise ValueError('Routing probability should be between 0 and 1.')
                cdf += col
            if cdf > 1:
                raise ValueError('Sum of routing probabilities in a row should be less than 1.')

    def num_jobs(self):
        """
        Returns the total number of jobs in the network.
        """

        num = 0
        for server in self.servers:
            num += server.num_jobs()
        return num

    def simulate(self, num_seconds, time_delta):
        """
        Simulates the network for a given number of seconds, progressing by time delta in each step.
        """

        time = 0
        while time < num_seconds:
            for server in self.servers:
                server.run(time_delta)
            #for server in self.servers:
            #    print(server.num_jobs(), end=' ')
            #print()
            time += time_delta

logging.basicConfig(format='%(message)s', level=logging.DEBUG)

R = [1, 1, 1, 1]
S = [1.25, 1.25, 1.25, 1.25]
P = [
    [0.21, 0.22, 0.23, 0.24],
    [0.8, 0.05, 0.04, 0.03],
    [0.9, 0.03, 0.04, 0.02],
    [0.85, 0.02, 0.03, 0.01],
]

N = Network(R, S, P)
N.simulate(50, 0.01)