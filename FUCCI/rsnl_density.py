import multiprocessing as mp
import os
import pickle as pkl
from functools import partial

import arviz as az
import jax.numpy as jnp
import jax.random as random
import matlab.engine
import numpyro
import numpyro.distributions as dist
import scipy.io as sio
from rsnl.inference import run_rsnl
from rsnl.model import get_robust_model
from rsnl.utils import engines
from rsnl.visualisations import plot_and_save_all


def bvcbm_simulation(sim_key, p1, p2, p3, m1, m2, m3, eng=None):
    theta = jnp.array([p1, p2, p3, m1, m2, m3])
    n = 1
    eng = engines[mp.current_process().pid]
    theta_matlab = matlab.double(theta.tolist())
    n_matlab = matlab.int64(n)
    len_obs_matlab = matlab.int64(15)
    sx_all = eng.simulator_density(theta_matlab, n_matlab, len_obs_matlab,  nargout=1)
    sx_all = jnp.asarray(sx_all)
    print(sx_all)
    return sx_all.flatten()


def sum_fn(x):
    return x


def get_prior():
    prior = dist.Uniform(
                low=jnp.array([0., 0., 0., 0., 0., 0.]),
                high=jnp.array([1., 1., 1., 10., 10., 10.])
            )
    return prior


def run_bvcbm():
    folder_name = "res/rsnl/real_density/"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    rng_key = random.PRNGKey(0)
    true_params = jnp.array([0.01, 0.15, 0.2, 5, 9, 5])

    model = get_robust_model
    rng_key = random.PRNGKey(0)
    prior = get_prior()

    sim_fn = partial(bvcbm_simulation)
    file_name = 'FUCCI/CellDensity_synthetic_dataset.mat'
    x_sim = sio.loadmat(file_name)['sy'][0]
    print('x_sim: ', x_sim)

    theta_dims = 6

    mcmc = run_rsnl(model,
                    prior,
                    sim_fn,
                    sum_fn,
                    rng_key,
                    x_sim,
                    jax_parallelise=False,
                    mp_parallelise=True,
                    num_cpus=6,
                    true_params=true_params,
                    theta_dims=theta_dims,
                    num_sims_per_round=10_000,
                    num_rounds=10,
                    save_each_round=True,
                    folder_name=folder_name
                    )

    mcmc.print_summary()
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}rsnl_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}rsnl_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)


if __name__ == '__main__':
    numpyro.set_host_device_count(6)
    run_bvcbm()
