function x = simulator(param,sim_params,length_of_simulation)

x = zeros(1,length_of_simulation);

x_temp = clib.Model.FullSimulation_biphasic(param(1), param(2), param(3), param(4), sim_params.page, param(6), param(7), param(8), param(9), param(5), sim_params.startingvol, length_of_simulation);

for i = 1:length_of_simulation
    x(i) = x_temp(i);

end