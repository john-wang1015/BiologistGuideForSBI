function [theta] = para_back_transformation(theta_tilde,len_y)

% computes back transformation of parameter for toad example
% INPUT:
% theta_tilde - parameter on the transformed space
% OUTPUT:
% theta - parameter on original space

e_theta_tilde = exp(theta_tilde);

a = [2, 1, 2];
b = [(len_y-1)*24, len_y-1, (len_y-1)*24];
% back transform
theta = a ./ (1 + e_theta_tilde) + b ./ (1 + 1 ./ e_theta_tilde);

end