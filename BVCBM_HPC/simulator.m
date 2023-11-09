function x = simulator(param,length_of_simulation)

starting_vol = double(100);
page = int32(2);

x = zeros(1,length_of_simulation);

x_temp = clib.Model.FullSimulation_biphasic(param(1), param(2), int32(param(3)), int32(param(4)), int32(page), param(6), param(7), int32(param(8)), int32(param(9)), param(5), starting_vol, int32(length_of_simulation));

for i = 1:length_of_simulation
    x(i) = x_temp(i);

end
