import elfi
import numpy as np
import matlab.engine
import os
from elfi.methods.bsl import pre_sample_methods, pdf_methods
from scipy.io import loadmat
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

def bvcbm_simulation(p1, p2, p3, m1, m2, m3, eng=None, batch_size=1, random_state=None):
    theta = np.array([p1, p2, p3, m1, m2, m3])

    sx_all = np.zeros((batch_size, 15))

    eng = matlab.engine.start_matlab()
    theta_matlab = matlab.double(theta.tolist())
    #sx_all = eng.simulator_tracking_nonparall(theta_matlab, 6,  nargout=1)
    for batch_ii in range(batch_size):
        sx = eng.simulator_density_nonparallp(theta_matlab, 15,  nargout=1)
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
    if true_params is None:
        true_params = np.array([0.4, 0.17, 0.08, 4.0, 4.0, 4.0])

    eng = matlab.engine.start_matlab()
    sim_fn = partial(bvcbm_simulation, eng=eng)
    y = sim_fn(*true_params)

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
    folder_name = "res/bsl/test_density"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    m = get_model()
    #likelihood = pdf_methods.standard_likelihood()
    feature_names = ['log_S']
    seed = 1
    # print("cores: ", multiprocessing.cpu_count())
    true_params = np.array([0.4, 0.17, 0.08, 4.0, 4.0, 4.0])
    print(true_params)

    #n_sim = 10000
    #true_params = {'r1': 0.4, 'r2': 0.17, 'r3': 0.08, 'm1':4.0, 'm2': 4.0, 'm3': 4.0}
    #pre_sample_methods.plot_features(m, true_params, n_sim, feature_names,
    #                                 seed=seed)
    #plt.savefig('rbslv_scim_features.png')

    #plt.clf()
    n_sim = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    log_stdev = pre_sample_methods.log_SL_stdev(model=m,
                             theta=true_params,
                             n_sim=n_sim,
                             feature_names=feature_names,
                             #likelihood=likelihood,
    #                         M=20,
                             seed=123)
    print('log_stdev: ', log_stdev)
    '''
    #print('stage 1 ready....')

    #nsim_round = 400
    #r_bsl_v = elfi.BSL(m,
    #                   nsim_round,
    #                   batch_size=50,
    #                   feature_names=feature_names,
    #                   likelihood=likelihood,
    #                   seed=seed
    #                   )
    #mcmc_iterations = 2
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

    params0 = [0.4, 0.17, 0.08, 4.0, 4.0, 4.0]
    res = r_bsl_v.sample(mcmc_iterations,
                         np.eye(6),
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

    res.plot_traces()
    plt.savefig('rbslv_bvcbm_traces.png')

    mbins = 30
    res.plot_marginals(reference_value=true_params,
                       bins=mbins)
    plt.savefig('rbslv_bvcbm_marginals.png')

    res.plot_pairs(reference_value=true_params,
                   bins=mbins)
    plt.savefig('rbslv_bvcbm_pairs.png')

    gamma_dict = dict(zip(['gamma_{}'.format(index) for index in range(r_bsl_v.observed.size)],
                          np.transpose(res.outputs['gamma'])))
    elfi.visualization.visualization.plot_marginals(gamma_dict, bins=mbins)
    plt.savefig('rbslv_bvcbm_gamma.png')

    '''

if __name__ == '__main__':
    run_bsl()