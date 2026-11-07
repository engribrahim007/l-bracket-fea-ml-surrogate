# My first Latin Hypercube Sampling: 2 inputs, 5 samples

from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=2, rng=1)
unit_samples = sampler.random(n=5)

print("Samples between 0 and 1:")
print(unit_samples)

lower = [3, 70]      # minimum: thickness (mm), leg length (mm)
upper = [8, 120]     # maximum: thickness (mm), leg length (mm)

real_samples = qmc.scale(unit_samples, lower, upper)

print("Real bracket values (thickness, leg length):")
print(real_samples)