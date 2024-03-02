"""
Module for simulation of open queueing networks.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

class Job:
    """
    Tracks the number of visits and the time spent by a job in a server.
    """

    NUM_JOBS = 0
    NUM_VISITS = []
    TIME_SPENT = []

    @classmethod
    def __init_class__(cls, num_servers):
        """
        Initializes the class variables used for logging.
        """
        for _ in range(num_servers):
            Job.NUM_VISITS.append(0)
            Job.TIME_SPENT.append(0.0)

    def __init__(self):
        """
        Initializes a new job.
        """

        self.num_visits = [ 0 for _, _ in enumerate(Job.NUM_VISITS) ]
        self.time_spent = [ 0.0 for _, _ in enumerate(Job.TIME_SPENT) ]

    def update(self, server, time):
        """
        Records the time spent in a new visit to a server.
        """

        self.num_visits[server] += 1
        self.time_spent[server] += time

    def wait(self, server, time):
        """
        Records the variable waiting time in a server.
        """

        self.time_spent[server] += time

    def leave(self):
        """
        Called when the job leaves the network.
        """

        Job.NUM_JOBS += 1
        for i, _ in enumerate(Job.NUM_VISITS):
            Job.NUM_VISITS[i] += self.num_visits[i]
            Job.TIME_SPENT[i] += self.time_spent[i]

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
        self.wait_queued_jobs(time_delta)
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
            logging.debug('%.3f %d: Queued an external arrival', self.timestamp, self.name + 1)
            self.queue.append(Job())
            self.next_arrival = np.random.exponential(1 / self.external_arrival_rate)

    def wait_queued_jobs(self, time_delta):
        """
        Adds waiting time to all queued jobs.
        """

        for job in self.queue:
            job.wait(self.name, time_delta)

    def service_current_job(self, time_delta):
        """
        Checks if the exponential timer for current job is up, routing it as necessary.
        """

        if self.is_busy:
            if self.service_remaining > time_delta:
                self.service_remaining -= time_delta
            else:
                logging.debug('%.3f %d: Serviced a job', self.timestamp, self.name + 1)
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
                logging.debug('%.3f %d: Routed job to neighbour %d', self.timestamp, self.name + 1, i + 1)
                self.channels[i].queue.append(self.current_job)
                self.current_job = None
                return

        cdf_i += self.self_loop_probability
        if cdf <= cdf_i:
            logging.debug('%.3f %d: Routed job to self', self.timestamp, self.name + 1)
            self.queue.append(self.current_job)
            self.current_job = None
            return

        self.current_job.leave()
        self.current_job = None
        logging.debug('%.3f %d: Job left the system', self.timestamp, self.name + 1)

    def draw_job(self):
        """
        Draws a job from the queue, assuming the server is idle.
        """

        if self.queue:
            logging.debug('%.3f %d: Started servicing a job', self.timestamp, self.name + 1)
            self.is_busy = True
            self.current_job = self.queue.pop(0)
            self.service_remaining = np.random.exponential(1 / self.service_rate)
            self.current_job.update(self.name, self.service_remaining)

class Network:
    """
    A network encapsulates all the servers, and provides a framework for synchronous simulation.
    """

    def __init__(self, arrival_rates, service_rates, routing_matrix, sampling_frequency):
        """
        Initializes the network for simulation.
        """

        self.validate_input(arrival_rates, service_rates, routing_matrix)

        self.arrival_rates = arrival_rates
        self.service_rates = service_rates
        self.routing_matrix = routing_matrix
        self.sample_threshold = max(round(1 / sampling_frequency, 0), 1)
        self.sample_counter = 0

        self.servers = []
        self.stats = {
            'num_samples': 0,
            'timestamps': [],
            'num_jobs_sum': [[] for i, _ in enumerate(self.routing_matrix)],
            'num_jobs_avg': [[] for i, _ in enumerate(self.routing_matrix)],
            'num_visits_avg': [[] for i, _ in enumerate(self.routing_matrix)],
            'time_spent_avg': [[] for i, _ in enumerate(self.routing_matrix)]
        }

        self.init_servers()

    def init_servers(self):
        """
        Initializes all the servers of the network.
        """
        for i, _ in enumerate(self.routing_matrix):
            server = Server(i, self.arrival_rates[i], self.service_rates[i])
            self.servers.append(server)

        for i, _ in enumerate(self.routing_matrix):
            self.servers[i].add_channels(
                self.servers[:i] + self.servers[i + 1:],
                self.routing_matrix[i][:i] + self.routing_matrix[i][i + 1:],
                self.routing_matrix[i][i]
            )

        Job.__init_class__(len(self.routing_matrix))

    def validate_input(self, arrival_rates, service_rates, routing_matrix):
        """
        Checks if rates are positive and routing matrix is a valid transition matrix.
        """
        if len(arrival_rates) < 1:
            raise ValueError('Number of servers should be positive')
        if len(arrival_rates) != len(service_rates) or len(service_rates) != len(routing_matrix) or len(routing_matrix) != len(routing_matrix[0]):
            raise ValueError('Number of servers is not consistent.')
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
            self.sample_counter += 1
            if self.sample_counter == self.sample_threshold:
                self.log_stats(time)
                self.sample_counter = 0

    def log_stats(self, current_time):
        """
        Logs the current statistics of the system.
        """

        self.stats['num_samples'] += 1
        self.stats['timestamps'].append(current_time)

        if self.stats['num_samples'] > 1:
            for i, _ in enumerate(self.routing_matrix):
                self.stats['num_jobs_sum'][i].append(
                    self.stats['num_jobs_sum'][i][-1] + self.servers[i].num_jobs()
                )
                self.stats['num_jobs_avg'][i].append(
                    self.stats['num_jobs_sum'][i][-1] / self.stats['num_samples']
                )
        else:
            for i, _ in enumerate(self.routing_matrix):
                self.stats['num_jobs_sum'][i].append(
                    self.servers[i].num_jobs()
                )
                self.stats['num_jobs_avg'][i].append(
                    self.stats['num_jobs_sum'][i][-1]
                )

        if Job.NUM_JOBS > 0:
            for i, _ in enumerate(self.routing_matrix):
                self.stats['num_visits_avg'][i].append(
                    Job.NUM_VISITS[i] / Job.NUM_JOBS
                )
                self.stats['time_spent_avg'][i].append(
                    Job.TIME_SPENT[i] / Job.NUM_JOBS
                )
        else:
            for i, _ in enumerate(self.routing_matrix):
                self.stats['num_visits_avg'][i].append(
                    0
                )
                self.stats['time_spent_avg'][i].append(
                    0.0
                )

    def plot_stats(self):
        """
        Plots the collected statistics.
        """

        print('Loading plots...')
        plt.figure(figsize=(10, 8))
        rows = len(self.routing_matrix) * 100

        for i, _ in enumerate(self.routing_matrix):
            plt.subplot(rows + 31 + 3*i)
            plt.plot(self.stats['timestamps'], self.stats['num_jobs_avg'][i])
            plt.title(f'Average number of jobs in server {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Number of jobs')

            plt.subplot(rows + 32 + 3*i)
            plt.plot(self.stats['timestamps'], self.stats['time_spent_avg'][i])
            plt.title(f'Average time spent in server {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Time spent per job')

            plt.subplot(rows + 33 + 3*i)
            plt.plot(self.stats['timestamps'], self.stats['num_visits_avg'][i])
            plt.title(f'Average number of visits to server {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Visits per job')

        plt.tight_layout()
        plt.show()

    def print_stats(self):
        """
        Prints the collected statistics.
        """

        print('_' * 35)
        print(f'NUM_JOBS = {Job.NUM_JOBS}')
        for i, _ in enumerate(self.routing_matrix):
            print()
            print(f"E[N_{i + 1}] = {self.stats['num_jobs_avg'][i][-1] :.2f}")
            print(f"E[T_{i + 1}] = {self.stats['time_spent_avg'][i][-1] :.2f}")
            print(f"E[V_{i + 1}] = {self.stats['num_visits_avg'][i][-1] :.2f}")
        print('_' * 35)

logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
