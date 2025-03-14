function plot_results_well_log(parameters, x, X, Y, Py, P, k)
% This function draws samples from the exact posterior distribution on the
% number of change points, their locations, and the parameters of the
% regression model, and then plots the results.
% Function differs from "plot_results" because of the underflow issue
% associated with larger data sets.  Notation is made within "plot_results"
% function where the necessary changes should be made

[N, m] =size(X);            % N = total # data points (N), m = # regressors
d_min = parameters(1);      % Minimum distance between adjacent change points    
k_0 = parameters(2);        % Hyperparameter for the prior on the regression coefficients
v_0 = parameters(3); sig_0 = parameters(4); % Hyperparameter for scaled inverse chi-square prior on the variance
k_max = parameters(5);      % Maximum number of change points
num_samp = parameters(6);   % Number of sampled solutions
beta0= zeros(m,1);          % Mean of multivariate normal prior on regression coefficients
I=eye(m);                   % m x m identity matrix


% Variables used in sampling procedure:
samp_holder=zeros(num_samp,k_max);  % Contains each of the num_samp change point solutions
chgpt_loc=zeros(1,N);               % The number of times a change point is identified at each data point
BETA = zeros(m,N);                  % Holds the regression coefficients

for i=1:num_samp
    %******** (i) Sample a Number of Change Points ***********************
    num_chgpts = pick_k1(k)-1;  % Since we allow for 0 changepoints, function returns the index of the 'k' vector, 
                                % which is offset from the number of change points by 1 
    if(num_chgpts>0)
        
        %******** (ii) Sample the Location of the Change Points ***********
        back=N;
        for kk=num_chgpts:-1:2      % Start at the end of the time series and work backwards
            temp=zeros(1,back-1);
            for v=1:back-1
                temp(v)= P(kk-1,v)+Py(v+1,back);  % Build the vector to sample from
            end
            M_temp = max(temp); temp = temp - M_temp;   % Avoids underflow 
            total=log(sum(exp(temp)));
            temp(:)=exp(temp(:)-total);  % Normalize the vector
            changepoint=pick_k1(temp);   % Sample the location of the change point
            chgpt_loc(changepoint)= chgpt_loc(changepoint) +1; % Keep track of change point locations
            samp_holder(i,kk)=changepoint;  % Keep track of individual change point solutions
            
            %******** (iii) Sample the Regression Parameters ***********
            % Regression Coefficients (Beta)
            XTy=X(changepoint+1:back,:)'*Y(changepoint+1:back); 
            J=k_0*I+X(changepoint+1:back,:)'*X(changepoint+1:back,:); 
            beta_hat=J\(k_0*beta0+XTy); %inv(J)*(k_0*beta0+XTy)
            
            for j=1:m
               BETA(j,changepoint+1:back) = BETA(j,changepoint+1:back) +beta_hat(j)*ones(1,back-changepoint);  
            end
            
            back=changepoint;   % Now work with the next segment
        end
        
        % The final changepoint
        %******** (ii) Sample the Location of the Change Points ***********
        kk=1;
        temp=zeros(1,back-1);
        for v=1:back-1
            temp(v)= Py(1,v)+Py(v+1,back); %Build the vector to sample from
        end
        M_temp = max(temp); temp = temp - M_temp;   % Avoids underflow
        total=log(sum(exp(temp)));
        temp(:)=exp(temp(:)-total); % Normalize the vector
        changepoint=pick_k1(temp);  % Sample the location of the change point
        chgpt_loc(changepoint)= chgpt_loc(changepoint) +1; % Keep track of change point locations
        samp_holder(i,kk)=changepoint;  % Keep track of individual change point solutions
            
        %******** (iii) Sample the Regression Parameters ***********
        % Regression Coefficients (Beta)
        XTy=X(changepoint+1:back,:)'*Y(changepoint+1:back); 
        J=k_0*I+X(changepoint+1:back,:)'*X(changepoint+1:back,:); 
        beta_hat=J\(k_0*beta0+XTy);     %inv(J)*(k_0*beta0+XTy)
        
        for j=1:m
               BETA(j,changepoint+1:back) = BETA(j,changepoint+1:back) +beta_hat(j)*ones(1,back-changepoint);  
        end
        
        %The final sub-interval
        XTy=X(1:changepoint,:)'*Y(1:changepoint); 
        J=k_0*I+X(1:changepoint,:)'*X(1:changepoint,:); 
        beta_hat=J\(k_0*beta0+XTy);     %inv(J)*(k_0*beta0+XTy)
        
        for j=1:m
               BETA(j,1:changepoint) = BETA(j,1:changepoint) +beta_hat(j)*ones(1,changepoint);  
        end
        
    else    % 0 change points, so a single homogeneous segment
        XTy=X'*Y; 
        J=k_0*I+X'*X; 
        
        %******** (iii) Sample the Regression Parameters ***********
        % Regression Coefficients (Beta)
        beta_hat=J\(k_0*beta0+XTy);     %inv(J)*(k_0*beta0+XTy)
        for j=1:m
               BETA(j,:) = BETA(j,:) +beta_hat(j)*ones(1,N);  
        end
    end
end

BETA=BETA/num_samp;             % Average regression coefficient at each data point.
chgpt_loc=chgpt_loc/num_samp;   % Posterior probability of a change point

%**********(iv) Plot the Results ********************************
% Adapt as Necessary

model=zeros(1,N);
for i=1:N
    model(i)=X(i,:)*BETA(:,i);
end

figure(1); plot(x,Y, 'LineWidth', 2); hold
title('The Well-Log Data Set', 'fontsize', 12)

[ax, h1, h2] = plotyy(x,model,x,chgpt_loc, 'plot');

gca=ax(1);
set(gca, 'Xlim', [0 4055])
set(gca, 'Xtick', 0:500:4050)
set(gca, 'Xticklabel', 0:500:4050)
%set(gca, 'Ylim', [-0.8 0.8])
%set(gca, 'Ytick', -0.8:0.2:0.8)
%set(gca, 'Yticklabel', -0.8:0.2:0.8)

set(gca,'Ycolor', 'k')      % Left Y-axis colored black
set(gca, 'fontsize', 12)    % Default is 10
set(get(ax(1), 'Ylabel'), 'String', 'Nuclear Response', 'fontsize', 12)
set(h1, 'Color', 'g')       % Model plotted in green, default blue
set(h1, 'LineStyle', '--'); % Default is a solid line
set(h1, 'LineWidth', 2)     % Default is 0.5

gca=ax(2);
set(gca, 'Xlim', [0 4055])
set(gca, 'Xtick', 0:500:4050)
set(gca, 'Xticklabel', 0:500:4050)
set(gca, 'Ylim', [0 3])
set(gca, 'Ycolor', 'r')     % Right Y-Axis colored red
set(gca, 'Yticklabel', 0:0.1:1)
set(gca, 'Ytick', 0:0.1:1)
set(gca, 'fontsize', 12)
set(get(ax(2), 'Ylabel'), 'String', 'Posterior Probability', 'fontsize', 12)
set(h2, 'Color', 'r')       % Posterior probabilities plotted in red
set(h2, 'LineWidth', 2)

line( [x(N); x(N)], [-2; 2], 'Color','k', 'LineWidth', 2);    
        %Draws a line representing the end of the data set


hold;

end