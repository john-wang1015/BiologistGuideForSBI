function x = simulator(param,length_of_simulation)

starting_vol = 100
page = 2

x = zeros(1,length_of_simulation);

x_temp = clib.Model.FullSimulation_biphasic(param(1), param(2), param(3), param(4), page, param(6), param(7), param(8), param(9), param(5), starting_vol, length_of_simulation);

for i = 1:length_of_simulation
    x(i) = x_temp(i);

end
