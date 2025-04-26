python execute.py --architecture=tanh_mlp_768_768_512_512_1 --dim=10 --batch_size=32 --strategy=hessian_trace --device=cpu --operator=laplacian --use_jax
python execute.py --architecture=tanh_mlp_768_768_512_512_1 --dim=10 --batch_size=32 --strategy=jet_naive --device=cpu --operator=laplacian --use_jax
python execute.py --architecture=tanh_mlp_768_768_512_512_1 --dim=10 --batch_size=32 --strategy=jet_simplified --device=cpu --operator=laplacian --use_jax
