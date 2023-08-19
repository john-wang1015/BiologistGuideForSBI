function theta = bayes_sl(y,N,M,n,cov_rw,sim_params1)
% bayes_sl_ricker_wood performs MCMC BSL on the Ricker example, using the same summary statistics as Wood (2010).
%
% INPUT:
% y - the observed data
% N - the starting population size (this will be 1 for our application)
% M - the number of iterations of BSL
% n - the number of simulated data sets SL estimation
% cov_rw - the covariance matrix of the random walk
%
% OUTPUT:
% theta - MCMC samples from the BSL target

theta_curr = [0.5 1e-4 40 350 25 0.5 1e-4 15 100];
T = length(y);

theta = zeros(M,9);
ssx = zeros(n,T);

% simulating n data sets
%parfor k = 1:n % for parallel computing
parfor i = 1:n
   ssx(i,:) = simulator(theta_curr,sim_params1,sim_params1.max_time);
end

the_mean = mean(ssx);
the_cov = cov(ssx);

% estimating the SL for current value
loglike_ind_curr = -0.5*log(det(the_cov)) - 0.5*(y-the_mean)*inv(the_cov)*(y-the_mean)';
        
for i = 1:M
    i
    %i  % print out iteration number if desired
    theta_prop = mvnrnd(theta_curr,cov_rw);
%     if (theta_prop(2) > 1e-3) ||  (theta_prop(7) > 1e-3)%sigma_e can't be negative
%         theta(i,:) = theta_curr;
%         continue;
%     end
    
	%simulating n data sets using the proposed parameters
    %parfor k = 1:n % for parallel computing
    parfor j = 1:n
        ssx(j,:) = simulator(theta_curr,sim_params1,sim_params1.max_time);
    end
    
    the_mean = mean(ssx);
    the_cov = cov(ssx);
    
	% estimating the SL for proposed value
    loglike_ind_prop = -0.5*log(det(the_cov)) - 0.5*(y-the_mean)*inv(the_cov)*(y-the_mean)';
    
    % MH accept-reject step
    if (exp(loglike_ind_prop - loglike_ind_curr) > rand)
        theta_curr = theta_prop;
        loglike_ind_curr = loglike_ind_prop;
    end
    theta(i,:) = theta_curr;
    
end

end