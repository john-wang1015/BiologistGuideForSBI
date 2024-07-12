clc;clear;
ncpus = 20; %number of CPUs to use
clust = parcluster('local');
clust.JobStorageLocation = tempdir;
par = parpool(clust,ncpus); 

load('CellDensity_synthetic_dataset.mat');

N = 10000;
sx = zeros(N,15);
computational_time_tracking = zeros(1,N);

InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");

parfor i = 1:N
    i
    sx(i,:) = simulator_density_nonparall(theta1, 15, InitPosData, CellTrackingData);
end

save("bsll_verification_density.mat")

figure
for i = 1:15
    subplot(3,5,i)
    histogram(log(sx(:,i)))
    xlabel(sprintf('\\theta_{%d}', i)); % Corrected the xlabel
    set(gca,'FontSize',24)
end

print(gcf, 'verfy_bsl_cell_density.png', '-dpng', '-r300');