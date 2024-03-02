from network import Network, Job

R = [0.5, 0]
S = [3, 3]
P = [
    [0.1, 0.9],
    [0, 0.2],
]

N = Network(R, S, P, 0.01)
N.simulate(15000, 0.001)

N.print_stats()
N.plot_stats()
