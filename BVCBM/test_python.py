import matlab.engine
import os
from rsnl.model import get_robust_model
from rsnl.inference import run_rsnl

import numpyro.distributions as dist
import jax.random as random
import jax.numpy as jnp


def sim_fn(sim_key, t1, t2, t3):
    eng = matlab.engine.start_matlab()
    bvcbm_path = os.path.abspath('BVCBM_HPC')
    model_path = os.path.join(bvcbm_path, 'Model')
    eng.addpath(bvcbm_path, nargout=0)
    eng.addpath(model_path, nargout=0)  # Explicitly add Model path
    #eng.defineModel(nargout=0)
    eng.test_run(nargout=0)
    eng.quit()


def run_bvcbm():
    rng_key = random.PRNGKey(0)
    test_theta = jnp.array([1.0, 1.0, 1.0])
    sim_fn(rng_key, *test_theta)
    pass


if __name__ == '__main__':
    run_bvcbm()
