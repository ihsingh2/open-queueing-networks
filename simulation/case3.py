from network import Network, Job

R = [20, 10, 0]
S = [30, 30, 50]
P = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0.1, 0]
]

N = Network(R, S, P, 0.01)
N.simulate(5000, 0.001)

N.print_stats()
N.plot_stats()
