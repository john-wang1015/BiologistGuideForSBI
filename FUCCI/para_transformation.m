function [theta_tilde] = para_transformation(theta)

% (1,2) by (0,100) by (0,0.9)

r1 = theta(:,1);
r2 = theta(:,2);
r3 = theta(:,3);

m1 = theta(:,4);
m2 = theta(:,5);
m3 = theta(:,6);

r1_tilde = log(r1 ./ (1-r1));
r2_tilde = log(r2 ./ (1-r2));
r3_tilde = log(r3 ./ (1-r3));

m1_tilde = log(m1 ./ (10-m1));
m2_tilde = log(m2 ./ (10-m2));
m3_tilde = log(m3 ./ (10-m3));

theta_tilde = [r1_tilde,r2_tilde,r3_tilde,m1_tilde,m2_tilde,m3_tilde];

end