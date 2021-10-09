function P = partition_fn(Py, k_max, N)
% The Forward Recursion step of the Sequential Bayesian Change Point Algorithm. 
% P(k,j) is the probability of the data Y_1:j containing k change points [Equation 2] 

P=zeros(k_max,N)-Inf;        % -Inf b/c starts in log form

%k is the number of change points
k=1;            % First row is different from the rest, as you add together two homogeneous segments

for j=k+1:N     % Note: Several of these terms will be -INF, due to d_min parameter
    temp=zeros(1,j-1);
    
    for v=1:j-1
        temp(v)= Py(1,v)+Py(v+1,j);     % Changepoints occur at start of new segment, not at end of old one
    end
    
    P(k,j) = log(sum(exp(temp)));       % Equation (2) - Marginalize over all possible placements of the change point.
    
    %NOTE: TO AVOID UNDERFLOW, USE:
    %{
    M_temp = max(temp);
    if (M_temp>-Inf)
        temp = temp - M_temp;
        P(k,j)=log(sum(exp(temp))) +M_temp;             % Equation (2) - Marginalize over all possible placements of the change point.
    else
        P(k,j) = -Inf;
    end
    %}
end

for k=2:k_max
    for j=(k+1):N  % Note: Several of these terms will be -INF as well

    temp=zeros(1,j-1);
    for v=1:j-1         
        temp(v) = P(k-1,v)+Py(v+1,j);
    end
    
    P(k,j) = log(sum(exp(temp)));       % Equation (2) - Marginalize over all possible placements of the change point.
    
    %NOTE: TO AVOID UNDERFLOW, USE:
    %{
    M_temp = max(temp);
    if (M_temp>-Inf)
        temp = temp - M_temp;
        P(k,j)=log(sum(exp(temp))) +M_temp;             % Equation (2) - Marginalize over all possible placements of the change point.
    else
        P(k,j) = -Inf;
    end
    %}
    
    end
    
end

end         % of function partition_fn 

