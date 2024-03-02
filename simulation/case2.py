from network import Network, Job

R = [1.5, 1]
S = [3, 5]
P = [
    [0, 0.33],
    [0.33, 0.33],
]

N = Network(R, S, P, 0.01)
N.simulate(15000, 0.001)

N.print_stats()
N.plot_stats()
