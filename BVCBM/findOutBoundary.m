function x = findOutBoundary(part_vals_prop,prior)
    if part_vals_prop(1) <= prior.alpha(1) || part_vals_prop(1) >= prior.beta(1)
        x = 0;
    elseif part_vals_prop(2) <= prior.alpha(1) || part_vals_prop(2) >= prior.beta(1)
        x = 0;
    elseif part_vals_prop(3) <= prior.lower(1) || part_vals_prop(3) >= prior.upper(1)
        x = 0;
    elseif part_vals_prop(4) <= prior.lower(2) || part_vals_prop(4) >= prior.upper(2)
        x = 0;
    elseif part_vals_prop(5) <= prior.tau_low || part_vals_prop(5) >= prior.tau_upper
        x = 0;
    elseif part_vals_prop(6) <= prior.alpha(1) || part_vals_prop(6) >= prior.beta(1)
        x = 0;
    elseif part_vals_prop(7) <= prior.alpha(1) || part_vals_prop(7) >= prior.beta(1)
        x = 0;
    elseif part_vals_prop(8) <= prior.lower(1) || part_vals_prop(8) >= prior.upper(1)
        x = 0;
    elseif part_vals_prop(9) <= prior.lower(2) || part_vals_prop(9) >= prior.upper(2)
        x = 0;
    else
        x= 1;
    end
    
     