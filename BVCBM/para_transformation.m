function [theta_tilde] = para_transformation(theta,len_y)

% (1,2) by (0,100) by (0,0.9)

a = theta(:,1);
b = theta(:,2);
c = theta(:,3);

r1_tilde = log((a - 2)./ (len_y*24-a));
r2_tilde = log((b-1) ./ (len_y-1-b));
r3_tilde = log((c-2) ./ (len_y*24-c));


theta_tilde = [r1_tilde,r2_tilde,r3_tilde];

end