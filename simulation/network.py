"""
Module for simulation of open queueing networks.
"""

import logging
import numpy as np

class Job:
    """
    Tracks the number of visits of a job to a server, and the total time spent in the network.
    """

    NUM_JOBS = 0
    NUM_VISITS = 0
    TIME_SPENT = 0

    def __init__(self):
        """
        Initializes a new job.
        """

        self.num_visits = 0
        self.time_spent = 0

    def update(self, time):
        """
        Records the time spent in a new visit to a server.
        """

        self.num_visits += 1
        self.time_spent += time

    def leave(self):
        """
        Called when the job leaves the network.
        """

        Job.NUM_JOBS += 1
        Job.NUM_VISITS += self.num_visits
        Job.TIME_SPENT += self.time_spent

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

        if self.external_arrival_rate == 0:
            self.external_arrival_rate = 0.00001
        if self.service_rate == 0:
            self.service_rate = 0.00001

        self.queue = []
        self.timestamp = 0
        self.is_busy = False
        self.current_job = None
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
            self.queue.append(Job())
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
                self.channels[i].queue.append(self.current_job)
                self.current_job = None
                return

        cdf_i += self.self_loop_probability
        if cdf <= cdf_i:
            logging.debug('%.2f %d: Routed job to self', self.timestamp, self.name)
            self.queue.append(self.current_job)
            self.current_job = None
            return

        self.current_job.leave()
        self.current_job = None
        logging.debug('%.2f %d: Job left the system', self.timestamp, self.name)

    def draw_job(self):
        """
        Draws a job from the queue, assuming the server is idle.
        """

        if self.queue:
            logging.debug('%.2f %d: Started servicing a job', self.timestamp, self.name)
            self.is_busy = True
            self.current_job = self.queue.pop(0)
            self.service_remaining = np.random.exponential(1 / self.service_rate)
            self.current_job.update(self.service_remaining)

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
        self.num_samples = 0
        self.num_jobs = [0 for i, _ in enumerate(self.routing_matrix)]

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
            if i < 0:
                raise ValueError('Arrival rates should be non-negative.')
        for i in service_rates:
            if i < 0:
                raise ValueError('Service rates should be non-negative.')
        for row in routing_matrix:
            cdf = 0
            for col in row:
                if col < 0 or col > 1:
                    raise ValueError('Routing probabilities should be between 0 and 1.')
                cdf += col
            if cdf > 1:
                raise ValueError('Sum of routing probabilities in a row should not exceed 1.')

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
            self.log()

    def log(self):
        """
        Logs the number of jobs in each server.
        """

        self.num_samples += 1
        for i, _ in enumerate(self.num_jobs):
            self.num_jobs[i] += self.servers[i].num_jobs()

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
