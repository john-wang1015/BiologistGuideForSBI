function [J] = jacobian_transformation (theta_tilde)

% computes jacobian of transformation for toad example
% INPUT:
% theta_tilde - parameter on the transformed space
% OUTPUT:
% J - jacobian value

e_theta_tilde = exp(theta_tilde);

a = [0, 0, 0, 0, 0, 0]; % lower bounds
b = [1, 1, 1, 10, 10, 10]; % upper bounds
logJ = log(b - a) - log(1 ./ e_theta_tilde + 2 + e_theta_tilde);
J = exp(sum(logJ));

end