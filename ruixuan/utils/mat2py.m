function [chgpt_loc, k] = mat2py(trajectory)

Y_complete=trajectory';
% Y_complete=Y_complete(10:end)';
x_complete = (1:length(Y_complete))';

N = length(Y_complete);                % If you want to look at entire data set at once, 
                % set N = length(Y_complete)
k_max=2;        % Maximum number of change points
d_min=5;        % Minimum distance between adjacent change points
      
k_0=0.01;           % Hyperparameter for the prior on the regression coefficients
v_0=1; sig_0=0.01;  % Hyperparameter for scaled inverse chi-square prior on the variance

Y = Y_complete(1:N);        % Work with initial data set only
x = x_complete(1:N);
X = [zeros(N,1)+1 (1:N)'];  % Build the regression model - in this case, a linear model
                            % Column 1 represents the constant term
                            % Column 2 represents the linear trend

[~, m] =size(X);            % r = total # data points (N), m = # regressors
beta0= zeros(m,1);          % Mean of multivariate normal prior on regression coefficients
num_samp=5000;               % Number of sampled solutions

parameters = [d_min k_0 v_0 sig_0 k_max num_samp];

%******* INITIAL DATA SET ******************
%******(3a)Calculate the Probability Density of Data for Each Sub-Interval*********
Py=zeros(N,N)-Inf;      % Matrix containing the probability of each substring of the data
I=eye(m);               % m x m identity matrix

for i=1:N
    for j=i+d_min-1:N
        
        %Calculate beta_hat
        XTy=X(i:j,:)'*Y(i:j); 
        J=k_0*I+X(i:j,:)'*X(i:j,:); 
        beta_hat=J\(k_0*beta0+XTy); %inv(J)*(k_0*beta0+XTy)
        
        %Calculate v_n and s_n
        a=j-i+1;    % 'a' is the number of data points.... not the distance between them, 
                    % which becomes relevant if data points are not equally spaced
        v_n=v_0+a;  % a is the number of data points in the segment, see above for definition
        
        s_n = v_0*sig_0 + k_0*(beta0-beta_hat)'*(beta0-beta_hat) + (Y(i:j,:)-X(i:j,:)*beta_hat)'*(Y(i:j,:)-X(i:j,:)*beta_hat);
        
        % Calculate the Probability Density of the Data - Equation (2) of Ruggieri 2013 - stored in log form 
        Py(i,j) = v_0*log(sig_0*v_0/2)/2 + gammaln(v_n/2) +m*log(k_0)/2  ...
            - log(gamma(v_0/2)) - v_n*log(s_n/2)/2 - a*log(2*pi)/2 - log(det(J))/2 ;    
     end
end

%****************(3b) Build Partition Function *************************

P=partition_fn(Py, k_max, N);

%****************(3c) Stochastic Backtrace via Bayes Rule****************
k= P(:,N);
for i=1:k_max                   % Equation (3)
    if(N-(i+1)*d_min+i >= i)    % If a plausible solution exists...
        k(i) = k(i) + log(0.5) - log(k_max) - log(nCk(N-(i+1)*d_min+i,i));
                                                % Above term is N_k
    end
end
k=[Py(1,N)+ log(0.5); k];       % Zero change points

total_k = log(sum(exp(k)));     % Adds logarithms and puts answer in log form 
k(:)=exp(k(:)-total_k);         % Normalize the vector - Used in Equations (4) and (7)

chgpt_loc = mat2py_plot(parameters, x, X, Y, Py, P, k);

% Eliminate unnecessary variables
clear i j J XTy beta_hat changepoint a back kk I num_chgpts r rr s_n v_n temp total_k v total new_x new_Y

end
