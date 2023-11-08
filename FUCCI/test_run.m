clc;clear;

CellTracking = true; % using cell tracking as summary statistics

theta1 = [0.04,0.17,0.08,4,4,4];
theta2 = [0.25,0.15,0.22,4,4,4];
theta3 = [0.12,0.07,0.03,4,4,4];
theta4 = [0.3,0.36,0.28,4,4,4];

sy1 = simulator(CellTracking, theta1);
sy2 = simulator(CellTracking, theta2);
sy3 = simulator(CellTracking, theta3);
sy4 = simulator(CellTracking, theta4);

save("CellTracking_synthetic_dataset.mat")

%%
clc;clear;

CellTracking = false; % using cell tracking as summary statistics

theta1 = [0.04,0.17,0.08,4,4,4];
theta2 = [0.25,0.15,0.22,4,4,4];
theta3 = [0.12,0.07,0.03,4,4,4];
theta4 = [0.3,0.36,0.28,4,4,4];

sy1 = simulator(CellTracking, theta1);
sy2 = simulator(CellTracking, theta2);
sy3 = simulator(CellTracking, theta3);
sy4 = simulator(CellTracking, theta4);

save("CellDensity_synthetic_dataset.mat")