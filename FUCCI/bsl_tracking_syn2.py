import elfi
import numpy as np
import matlab.engine
import os
import scipy.io as sio
from elfi.methods.bsl import pre_sample_methods, pdf_methods
from scipy.io import loadmat,savemat
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

def bvcbm_simulation(p1, p2, p3, m1, m2, m3, eng=None, batch_size=1, random_state=None):
    theta = np.array([p1, p2, p3, m1, m2, m3])

    sx_all = np.zeros((batch_size, 6))

    eng = matlab.engine.start_matlab()
    theta_matlab = matlab.double(theta.tolist())
    #sx_all = eng.simulator_tracking_nonparall(theta_matlab, 6,  nargout=1)
    for batch_ii in range(batch_size):
        sx = eng.simulator_tracking_nonparall(theta_matlab, 6,  nargout=1)
        sx = np.array(sx).flatten()
        sx_all[batch_ii, :] = sx

    eng.quit()
    #sx_all = np.asarray(sx_all)
    #print(theta)

    print(sx_all.shape)
    return sx_all

def sum_fn(x):
    """Identity."""
    return x

def log_sum_fn(x):
    return np.log(x)

def get_model(true_params=None, x_obs=None):
    #if true_params is None:
    #    true_params = np.array([0.4, 0.17, 0.08, 4.0, 4.0, 4.0])

    eng = matlab.engine.start_matlab()
    sim_fn = partial(bvcbm_simulation, eng=eng)
    y = x_obs#sim_fn(*true_params)

    m = elfi.ElfiModel()
    # prior
    elfi.Prior('uniform', 0, 1, model=m, name='r1')
    elfi.Prior('uniform', 0, 1, model=m, name='r2')
    elfi.Prior('uniform', 0, 1, model=m, name='r3')

    elfi.Prior('uniform', 0, 10, model=m, name='m1')
    elfi.Prior('uniform', 0, 10, model=m, name='m2')
    elfi.Prior('uniform', 0, 10, model=m, name='m3')
    # simulator
    elfi.Simulator(sim_fn, m['r1'], m['r2'],  m['r3'], m['m1'], m['m2'],  m['m3'], observed=y, name='cell')
    # summary
    elfi.Summary(sum_fn, m['cell'], name='S')
    elfi.Summary(log_sum_fn, m['cell'], name='log_S')
    # distance
    elfi.Distance('euclidean', m['S'], name='d')
    return m

def run_bsl():
    folder_name = "res/bsl/syn2_tracking"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    x_obs = sio.loadmat('CellTracking_synthetic_dataset.mat')['sy2']
    true_params = sio.loadmat('CellTracking_synthetic_dataset.mat')['theta2']

    m = get_model(true_params, x_obs)
    likelihood = pdf_methods.standard_likelihood()
    feature_names = ['log_S']
    seed = 1

    print('stage 1 ready....')

    nsim_round = 800
    r_bsl_v = elfi.BSL(m,
                       nsim_round,
                       batch_size=400,
                       feature_names=feature_names,
                       likelihood=likelihood,
                       seed=seed
                       )
    mcmc_iterations = 10000
    #est_posterior_cov = loadmat('cov_matrix_subset.mat')['subset_matrix']
    #est_posterior_cov = np.array(est_posterior_cov).reshape((3,3))
    logit_transform_bound = [(0, 1),
                             (0, 1),
                             (0, 1),
                             (0, 10),
                             (0, 10),
                             (0, 10)]

    elfi.set_client('multiprocessing')

    print('start to sample ... ')

    params0 = true_params.tolist()[0]
    res = r_bsl_v.sample(mcmc_iterations,
                         0.1*np.eye(6),
                         #est_posterior_cov,
                         burn_in=1,
                         param_names=['r1','r2','r3','m1','m2','m3'],
                         params0=params0,
                         logit_transform_bound=logit_transform_bound,
                         bar=False
                         )
    print(res)

    print(res.compute_ess())

    with open(f'{folder_name}/bsl_theta.pkl', 'wb') as f:
        pkl.dump(res.samples, f)

    savemat(f'{folder_name}/bsl_theta.mat', {"posterior": res.samples})

    res.plot_traces()
    plt.savefig(f'{folder_name}/bslv_bvcbm_traces.png')

    mbins = 30
    res.plot_marginals(reference_value=true_params,
                       bins=mbins)
    plt.savefig(f'{folder_name}/bslv_bvcbm_marginals.png')

    res.plot_pairs(reference_value=true_params,
                   bins=mbins)
    plt.savefig(f'{folder_name}/bslv_bvcbm_pairs.png')

    gamma_dict = dict(zip(['gamma_{}'.format(index) for index in range(r_bsl_v.observed.size)],
                          np.transpose(res.outputs['gamma'])))
    elfi.visualization.visualization.plot_marginals(gamma_dict, bins=mbins)
    plt.savefig(f'{folder_name}/bslv_bvcbm_gamma.png')

    

if __name__ == '__main__':
    run_bsl()