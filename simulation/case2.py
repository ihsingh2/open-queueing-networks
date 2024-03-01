from network import Network, Job

R = [1.5, 1]
S = [3, 5]
P = [
    [0, 0.33],
    [0.33, 0.33],
]

N = Network(R, S, P)
N.simulate(15000, 0.001)

print()

print(f'NUM_JOBS = {Job.NUM_JOBS}')
print(f'SUM[NUM_VISITS] = {Job.NUM_VISITS}')
print(f'SUM[TIME_SPENT] = {Job.TIME_SPENT :.2f}')
print(f'E[NUM_VISITS] = {Job.NUM_VISITS / Job.NUM_JOBS :.2f}')
print(f'E[TIME_SPENT] = {Job.TIME_SPENT / Job.NUM_JOBS :.2f}')

print()

for id, jobs in enumerate(N.num_jobs):
    print(f'E[N_{id}] = {jobs / N.num_samples :.2f}')
