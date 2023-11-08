clc;clear;
addpath("BVCBM_HPC/Model/");
total_time = 32;

prior.num_params = 9; 
prior.alpha = [0, 0];
prior.beta = [1, 1e-5];
prior.lower = [0, 1];
prior.upper = [50, total_time*24];
prior.tau_low = 2;
prior.tau_upper = total_time-1;

prior.sampler = @() [unifrnd(prior.alpha, prior.beta),...
    unifrnd(prior.lower, prior.upper),...
    floor(unifrnd(prior.tau_low, prior.tau_upper)),...
    unifrnd(prior.alpha, prior.beta),...
    unifrnd(prior.lower, prior.upper)];

sim_params.page = 2;
sim_params.max_time = 32;
sim_params.startingvol = 100;

part_vals = prior.sampler()
part_sim = simulator(part_vals, sim_params, total_time)
