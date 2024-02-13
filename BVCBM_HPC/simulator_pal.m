function x = simulator(param,length_of_simulation)

starting_vol = double(100);
page = int32(2);
nsize = size(param);

x = zeros(nsize(1),length_of_simulation);

parfor j = 1:nsize(1)
    x_temp = clib.Model.FullSimulation_biphasic(param(j,1), param(j,2), int32(param(j,3)), int32(param(j,4)), int32(page), param(j,6), param(j,7), int32(param(j,8)), int32(param(j,9)), param(j,5), starting_vol, int32(length_of_simulation));

    for i = 1:length_of_simulation
        x(j,i) = x_temp(i);
    end
end