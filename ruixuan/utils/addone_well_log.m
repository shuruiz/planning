function [Py, P] = addone_well_log(parameters, X, Y, Py, P)
% Input: Updated versions of X and Y; Original versions of Py and P.
% Output: Updated versions of Py and P.


[N, m] =size(X);            % N = total # data points (N), m = # regressors
d_min = parameters(1);      % Minimum distance between adjacent change points    
k_0 = parameters(2);        % Hyperparameter for the prior on the regression coefficients
v_0 = parameters(3); sig_0 = parameters(4); % Hyperparameter for scaled inverse chi-square prior on the variance
k_max = parameters(5);      % Maximum number of change points
num_samp = parameters(6);   % Number of sampled solutions
beta0= zeros(m,1);          % Mean of multivariate normal prior on regression coefficients
I=eye(m);                   % m x m identity matrix

Py_temp = zeros(N,1) -Inf;


%***************** Calculate the Probabilities **********************
for i=1:N-d_min+1
        if (N - i <1500)    % this should speed things up a bit - all change points are within 1500 of each other
        %Calculate beta_hat
        XTy=X(i:N,:)'*Y(i:N); 
        J=k_0*I+X(i:N,:)'*X(i:N,:); 
        beta_hat=J\(k_0*beta0+XTy);     %inv(J)*(k_0*beta0+XTy)
        
        %Calculate v_n and s_n
        a=N-i+1;    % 'a' is the number of data points.... not the distance between them, 
                    % which becomes relevant if data points are not equally spaced
        v_n=v_0+a;  % a is the number of data points in the segment, see above for definition
        
        s_n = v_0*sig_0 + k_0*(beta0-beta_hat)'*(beta0-beta_hat) + (Y(i:N,:)-X(i:N,:)*beta_hat)'*(Y(i:N,:)-X(i:N,:)*beta_hat);
        
        % Calculate the Probability Density of the Data - Equation (2) of Ruggieri 2013 - stored in log form 
        Py_temp(i) = v_0*log(sig_0*v_0/2)/2 + gammaln(v_n/2) +m*log(k_0)/2  ...
            - log(gamma(v_0/2)) - v_n*log(s_n/2)/2 - a*log(2*pi)/2 - log(det(J))/2 ;    %Minus because J is no longer inverted!
        end
end

%Update of Py matrix
Py = [Py Py_temp(1:N-1); zeros(1,N)-Inf];

%*********** Update the Partition Function ********************
P_temp=zeros(k_max,1)-Inf;        % -Inf b/c starts in log form

%k is the number of Change Points
k=1;            % First row is different from the rest, as you add together two homogeneous segments

    temp=zeros(1,N-1);
    for v=1:N-1
        temp(v)= Py(1,v)+Py(v+1,N);     % Changepoints occur at start of new segment, not at end of old one
    end
    M_temp = max(temp);                 % Corrects potential underflow issues
    if (M_temp>-Inf)
        temp = temp - M_temp;
        P_temp(k)=log(sum(exp(temp))) +M_temp;             % Equation (2) - Marginalize over all possible placements of the change point.
    else
        P_temp(k) = -Inf;
    end

for k=2:k_max
    temp=zeros(1,N-1);
    for v=1:N-1         
        temp(v) = P(k-1,v)+Py(v+1,N);
    end
    M_temp = max(temp);                 % Corrects potential underflow issues
    if (max(temp)>-Inf)
        temp = temp - M_temp;
        P_temp(k)=log(sum(exp(temp))) + M_temp;     % Equation (2) - Marginalize over all possible placements of the change point.
    else
        P_temp(k) = -Inf;
    end
end

%Update of P matrix
P = [P P_temp];

end