function [J] = jacobian_transformation (theta_tilde,len_y)

% computes jacobian of transformation for toad example
% INPUT:
% theta_tilde - parameter on the transformed space
% OUTPUT:
% J - jacobian value

e_theta_tilde = exp(theta_tilde);

a = [2, 1, 2];
b = [(len_y-1)*24, 31, (len_y-1)*24];

logJ = log(b - a) - log(1 ./ e_theta_tilde + 2 + e_theta_tilde);
J = exp(sum(logJ));

end