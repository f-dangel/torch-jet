python execute.py --architecture=tanh_mlp_768_768_512_512_1 --dim=5 --batch_size=256 --strategy=jet_simplified --device=cpu --operator=bilaplacian

# Before re-arranging the graph:
# Bilaplacian non-differentiable (jet_simplified): 8.07e-01 GiB
# Bilaplacian (jet_simplified): 3.08e+00 GiB
# Bilaplacian (jet_simplified): 1.77361 ± 0.15226 s (best: 1.65449 s)

# After ASAP scheduling
# Bilaplacian non-differentiable (jet_simplified): 1.02e+00 GiB
# Bilaplacian (jet_simplified): 2.49e+00 GiB
# Bilaplacian (jet_simplified): 1.70289 ± 0.03999 s (best: 1.65792 s)
