function [theta, loglike] = bsl_cell_tracking(y,init,m,M,cov_rw,prior,file_name)
ncpus = 16; %number of CPUs to use
clust = parcluster('local');
clust.JobStorageLocation = tempdir;
par = parpool(clust,ncpus); 

InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");

init = init';
x = zeros(m,length(y));
parfor (k = 1:m,ncpus)
    x(k,:) = simulator_tracking_nonparall(init, 6, InitPosData, CellTrackingData);
end

%Calculating the mean and covariance of the summary statistics
the_mean = mean(x);
the_cov = cov(x);
L = chol(the_cov);
logdetA = 2*sum(log(diag(L)));

% synthetic likelihood
loglike_curr = -0.5*logdetA - 0.5*(y-the_mean)/the_cov*(y-the_mean)';


theta = zeros(M,prior.num_params);
loglike = zeros(M,1);

theta_curr = prior.trans_f(init);

for i = 1:M
    i
    theta_prop = mvnrnd(theta_curr,cov_rw);
    
    logprior_curr = log(prior.pdf(theta_curr));
    logprior_prop = log(prior.pdf(theta_prop));
    
    prop = prior.trans_finv(theta_prop);

    ssx = zeros(m,length(y));
    parfor (k = 1:m,ncpus)
        ssx(k,:) = simulator_tracking_nonparall(prop, 6, InitPosData, CellTrackingData);
    end
    if (any(any(isnan(ssx))))
        theta(i,:) = theta_curr;
        loglike(i) = loglike_curr;
        continue;
    end
        
    
    %Calculating the mean and covariance of the summary statistics
    the_mean = mean(ssx);
    the_cov = cov(ssx);
    [L,p] = chol(the_cov);
    if (p>0)
        theta(i,:) = theta_curr;
        loglike(i) = loglike_curr;
        continue;
    end
        
    logdetA = 2*sum(log(diag(L)));
    
    % synthetic likelihood
    loglike_prop = -0.5*logdetA - 0.5*(y-the_mean)/the_cov*(y-the_mean)';

    mh = exp(loglike_prop - loglike_curr + logprior_prop - logprior_curr);
    
    if (mh > rand)
        fprintf('**** accept ****\n');
        theta_curr = theta_prop;
        
        prior.trans_finv(theta_curr)

        loglike_curr = loglike_prop;
    end
    theta(i,:) = theta_curr;
    loglike(i) = loglike_curr;
    
    if mod(i,500) == 0
        theta_temp = zeros(i,prior.num_params);
        for j = 1:i
            theta_temp(j,:) = prior.trans_finv(theta(j,:));
        end
        str1 = append(file_name, num2str(i));
        save_file_name = append(str1, '.mat');

        save(save_file_name,'theta_temp');
    end

end

% back transform
for i=1:M
    theta(i,:) = prior.trans_finv(theta(i,:));
end


end
