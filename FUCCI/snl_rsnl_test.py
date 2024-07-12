import matlab.engine
import numpy as np
import os
import arviz as az
from rsnl.model import get_standard_model
from rsnl.inference import run_snl
from rsnl.visualisations import plot_and_save_all
import scipy.io as sio
import jax
from jax import device_get

import numpyro.distributions as dist
import jax.random as random
import jax.numpy as jnp
import pickle as pkl
from functools import partial

def theta_convert(theta):
    return theta

def bvcbm_simulation(sim_key, p1, p2, p3, m1, m2, m3, eng=None):

    if hasattr(p1, 'shape'):
        p1 = p1.flatten()[0]
        p2 = p2.flatten()[0]
        p3 = p3.flatten()[0]
        m1 = m1.flatten()[0]
        m2 = m2.flatten()[0]
        m3 = m3.flatten()[0]

    theta = jnp.array([p1, p2, p3, m1, m2, m3])
    print(device_get(theta[0]).tolist())

    eng = matlab.engine.start_matlab()
    theta_matlab = matlab.double(device_get(theta).tolist())
    sx_all = eng.simulator_tracking_nonparall(theta_matlab, 6,  nargout=1)
    eng.quit()
    sx_all = jnp.asarray(sx_all)
    print(sx_all.flatten())
    print("**********************************************")
    return sx_all.flatten()

def sum_fn(x):
    return x

def get_prior():
    prior = dist.Uniform(
                low=jnp.array([0., 0., 0., 0., 0., 0.]),
                high=jnp.array([1., 1., 1., 1., 1., 1.])
            )
    return prior

def run_bvcbm():
    folder_name = "res/snl/test"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)
        
    rng_key = random.PRNGKey(0)
    true_params = jnp.array([0.4, 0.17, 0.08, 0.4, 0.4, 0.4])
    
    model = get_standard_model
    rng_key = random.PRNGKey(0)
    prior = get_prior()
    eng = matlab.engine.start_matlab()
    sim_fn = partial(bvcbm_simulation, eng=eng)
    x_sim = sim_fn(rng_key, *true_params)
    #x_sim = jnp.array([sio.loadmat('CellTracking_synthetic_dataset.mat')['sy1']]).flatten()
    print('x_sim: ', x_sim)

    theta_dims = 6

    mcmc = run_snl(model,
                   prior,
                   sim_fn,
                   sum_fn,
                   rng_key,
                   x_sim,
                   jax_parallelise=True,
                   true_params=true_params,
                   theta_dims=6,
                   num_sims_per_round=500,
                   num_rounds=1
                   )

    mcmc.print_summary()
    inference_data = az.from_numpyro(mcmc)

    with open(f'{folder_name}/snl_thetas.pkl', 'wb') as f:
        pkl.dump(inference_data.posterior.theta, f)

    plot_and_save_all(inference_data, true_params, folder_name=folder_name)

if __name__ == '__main__':
    run_bvcbm()