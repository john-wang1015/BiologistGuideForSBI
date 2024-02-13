import elfi
import numpy as np
import matlab.engine
import os
from elfi.methods.bsl import pre_sample_methods, pdf_methods
import scipy.io as sio
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import torch

from sbi.inference import SNPE_A, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
_ = torch.manual_seed(0)

def bvcbm_simulation(theta):
    eng = matlab.engine.start_matlab()
    bvcbm_path = os.path.abspath('BVCBM_HPC')
    model_path = os.path.join(bvcbm_path, 'Model')
    eng.addpath(bvcbm_path, nargout=0)
    eng.addpath(model_path, nargout=0)  
    print(theta.shape)
    theta_matlab = matlab.double(theta.tolist())

    sx_all = np.zeros((len(theta.tolist()), 32))
    print(sx_all.shape)
    sx_all = eng.simulator_pal(theta_matlab, 32,  nargout=1)
    eng.quit()
    sx_all = np.asarray(sx_all)
    print(sx_all)
    return torch.from_numpy(sx_all).to(torch.float32)

def simulators(theta):
    return bvcbm_simulation(theta)

def run_snpe_c():
    x_0 = sio.loadmat("paper2_BVCBM_synthetic_datasets.mat")['sim3']

    prior_min = torch.as_tensor([0,0,17,2.0, 2.0,0,0,17, 2.0])
    prior_max = torch.as_tensor([1, 1e-4,18, 32*24-1.0, 31.0,1, 1e-4,18, 32*24-1.0])
    prior = utils.BoxUniform(low=prior_min, high=prior_max)
    simulator, prior = prepare_for_sbi(simulators, prior)

    inference = SNPE_A(prior=prior, density_estimator='mdn_snpe_a')

    num_rounds = 10

    posteriors = []
    posterior_samples = []
    proposal = prior

    for kk in range(num_rounds):
        print("*****************iteration {}**************************".format(kk))
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=2000, simulation_batch_size = 2000)

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)

        posterior_sample = posterior.sample((2000,), x=x_0)

        figure(figsize=(20, 5))
        for i in range(1,10):
            plt.subplot(3,3,i)
            sns.kdeplot(data=posterior_sample[:, i-1], color='blue')
            if i == 1:
                plt.legend(labels=['Proposal', 'NPE', 'True'])
        plt.savefig('snpa_bvcbm_syn3_iteration {}.png'.format(kk))
        plt.show()
        
        string2save = "snpe_a_nsf syn3, round {}.mat".format(kk)
        sio.savemat(string2save, {"posterior":posterior_sample})
        
        posterior_samples.append(posterior_sample)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_0)

    folder_name = "res/npea/syn3"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    # Iterate through each model in the list and save
    for idx, model in enumerate(posteriors):
        file_path = os.path.join(folder_name, f'model_{idx}.pth')
        torch.save(model.state_dict(), file_path)



if __name__ == '__main__':
    run_snpe_c()



