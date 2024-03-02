from network import Network, Job

R = [1, 0, 0]
S = [3, 5, 6]
P = [
    [0, 1, 0],
    [0, 0, 0.75],
    [0, 1, 0]
]

N = Network(R, S, P)
N.simulate(15000, 0.001)

N.print_stats()
N.plot_stats()
