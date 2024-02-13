import matlab.engine
import os
from rsnl.model import get_robust_model
from rsnl.inference import run_rsnl
from rsnl.visualisations import plot_and_save_all

import numpyro.distributions as dist
import jax.random as random
import jax.numpy as jnp
import arviz as az
import pickle as pkl
from functools import partial
import scipy.io as sio

def bvcbm_simulation(sim_key, g_age, tau, g_age_2, eng=None):

    t1 = 1.0
    t2 = 0.0
    t3 = 17.3
    t6 = 1.0
    t7 = 0.0
    t8 = 17.3

    theta = jnp.array([t1, t2, t3, g_age, tau, t6, t7, t8, g_age_2])

    length_of_simulation = 32
    start_matlab = False if eng is not None else True
    if start_matlab:
        eng = matlab.engine.start_matlab()
    bvcbm_path = os.path.abspath('BVCBM_HPC')
    model_path = os.path.join(bvcbm_path, 'Model')
    eng.addpath(bvcbm_path, nargout=0)
    eng.addpath(model_path, nargout=0)  # Explicitly add Model path
    #eng.defineModel(nargout=0)
    #eng.test_run(nargout=0)
    theta_matlab = matlab.double(theta.tolist())

    sx = eng.simulator(theta_matlab, length_of_simulation,  nargout=1)
    if start_matlab:
        eng.quit()

    return jnp.array(sx).flatten()

def sum_fn(x):
    return x

def get_prior():
    prior = dist.Uniform(
                low=jnp.array([2.0, 2.0, 2.0]),
                high=jnp.array([32*24-1.0, 31.0, 32*24-1.0])
            )
    return prior


def run_bvcbm():
    rng_key = random.PRNGKey(0)
    true_params = jnp.array([300.0, 16.0, 300.0])
    
    model = get_robust_model
    rng_key = random.PRNGKey(0)
    prior = get_prior()
    
    eng = matlab.engine.start_matlab()
    sim_fn = partial(bvcbm_simulation, eng=eng)
    #x_sim = sim_fn(rng_key, *true_params)
    pancreatic = sio.loadmat("CancerDatasets.mat")['Pancreatic_data']
    x_sim = pancreatic[0:32,2]
    print('x_sim: ', x_sim)

    theta_dims = 3

    mcmc = run_rsnl(model,
                    prior,
                    sim_fn,
                    sum_fn,
                    rng_key,
                    x_sim,
                    jax_parallelise=False,
                    true_params=true_params,
                    theta_dims=3,
                    num_sims_per_round=10000,
                    num_rounds=10
                    )

    mcmc.print_summary()
    inference_data = az.from_numpyro(mcmc)
    
    folder_name = 'res/rsnl/pan3/'

    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    with open(f'{folder_name}rsnl_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    with open(f'{folder_name}rsnl_adj_params.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.adj_params, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)


if __name__ == '__main__':
    run_bvcbm()
