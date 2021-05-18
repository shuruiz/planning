%************************************************************************/
%* The Sequential Bayesian Change Point algorithm - A program to        */
%* caluclate the posterior probability of a change point in a time      */
%* series.                                                              */
%*                                                                      */
%* Please acknowledge the program author on any publication of          */
%* scientific results based in part on use of the program and           */
%* cite the following article in which the program was described.       */
%*                                                                      */
%* E. Ruggieri and M. Antonellis.  "An exact approach to Bayesian       */
%* sequential change point detection"                                   */
%*                                                                      */
%* Program Author: Eric Ruggieri                                        */
%* College of the Holy Cross                                            */
%* Worcester, MA 01610                                                  */
%* Email:  eruggier@holycross.edu                                       */
%*                                                                      */
%* Copyright (C) 2016  College of the Holy Cross                        */
%*									                                    */
%* Acknowledgements: This work was supported by a grant from the        */
%* National Science Foundation, DMS-1407670 (E. Ruggieri, PI)           */
%*                                                                      */
%* The Sequential Bayesian Change Point algorithn is free software:     */
%* you can redistribute it and/or modify it under the terms of the GNU  */
%* General Public License as published by the Free Software Foundation, */
%* either version 3 of the License, or (at your option) any later       */
%* version.                                                             */
%*                                                                      */
%* The Sequential Bayesian Change Point algorithm is distributed in the */
%* hope that it will be useful, but WITHOUT ANY WARRANTY; without even  */ 
%* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  */
%* PURPOSE.  See the GNU General Public License for more details.       */
%*                                                                      */
%* You should have received a copy of the GNU General Public License    */
%* along with the Sequential Bayesian Change Point algorithm.  If not,  */
%* see <http://www.gnu.org/licenses/> or write to the                   */
%* Free Software Foundation, Inc.                                       */
%* 51 Franklin Street Fifth Floor                                       */
%* Boston, MA 02110-1301, USA.                                          */
%************************************************************************/
%
% Outline of the Sequential Bayesian Change Point algorithm:
% 1) Load the data  
% 2) Define the parameter values
% 3) For the initial data set:
%   a) Calculate the probability density of the data for each sub-interval
%   b) Forward Recursion [Dynamic Programming]
%   c) If a change point is detected:
%       Stochastic Backtrace via Bayes rule
%       i) Sample a number of change points
%       ii) Sample the location of the change points
%       iii) Sample the regression parameters between adjacent change points
%       iv) Plot the results
% 4) (As needed) Add additional data points one at a time to existing data set
% 5) Repeat steps 3a-3c for new observation
% 6) Plot the results
%
% Description of parameters, input, and output can be found in accompanying 
% ReadMe file
%

clear;

%************(1)Load the Data*****************************
load scene_0.mat;
% load Temperature_Anomaly2013.txt;

index=find((inf>AV_velocity) & (AV_velocity>0));
Y_complete=AV_velocity(index)';
% Y_complete=Y_complete(10:end)';
x_complete = (index(1):index(end))';
% Y_complete=Temperature_Anomaly2013(:,2);    % Data set / time series
% x_complete=Temperature_Anomaly2013(:,1);    % Time points

%*************(2)Define the Parameter Values**************

% N=10;           % Number of data points in initial data set.  
N = length(Y_complete);                % If you want to look at entire data set at once, 
                % set N = length(Y_complete)
k_max=round(N/50);        % Maximum number of change points
d_min=50;        % Minimum distance between adjacent change points
      
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

%If analyzing the complete data set all at once you may want to use the
%next line of code:
% plot_results(parameters, x, X, Y, Py, P, k);

%********************ADDING A NEW DATA POINT ***************************
% This part of the code loops through all remaining observations in the 
% data set.  If you are adding only a single observations, then you can 
% comment out the loop

% current_num=0;         % Assumes no change points have yet to be detected.  
% KK=zeros(k_max+1,N);   % A record of the posterior distribution of k as new observations are added

% for n = N+1:length(Y_complete)
%     % Define variables: new_x [Time Point], new_Y [Data Point] and append these
%     % to the existing x, and Y.  Also, update N and the regression model, X.
%     % Example - Taking the next data point in the temperature anomaly data set
%     new_Y = Y_complete(n);      Y = [Y; new_Y];
%     new_x = x_complete(n);      x = [x; new_x];
%     N = N+1;
%     X = [X; 1 N];       % Linear Model
%     
%     %Function call to update Py and P matrices:
%     [Py, P] = addone(parameters, X, Y, Py, P);
%     
%     
% % From here, you'll have to decide whether or not to place a new change
% % point.  One way could be to look at how the distribution on k (the posterior
% % distribution on the # change points) changes as you add more data points.
% 
%     k= P(:,N);
%     for i=1:k_max                   % Equation (3)
%         if(N-(i+1)*d_min+i >= i)    % If a plausible solution exists...
%             k(i) = k(i) + log(0.5) - log(k_max) - log(nCk(N-(i+1)*d_min+i,i));
%                                                     % Above term is N_k
%         end
%     end
%     k=[Py(1,N)+ log(0.5); k];       % Zero change points
% 
%     total_k = log(sum(exp(k)));     % Adds logarithms and puts answer in log form 
%     k(:)=exp(k(:)-total_k);         % Normalize the vector - Equations (4) and (7)
% 
% 
%     %**** Detecting Change Points *******
%     KK = [KK k];
%     
%     %Use the MODE
%     %[~, M] = max(k); M = M-1;        %adjusts for 0 change points
%     
%     
%     %Use the MEDIAN
%     temp=0; M = 0;
%     while(temp<0.5)
%     M=M+1;
%     temp = temp+k(M);
%     end
%     M=M-1;                          %adjusts for 0 change points
%     
%     %Use the MEAN
%     %{
%     temp=0;
%     for i = 1:length(k)
%         temp = temp+(i-1)*k(i);
%     end
%     M= round(temp);
%     %}
%     
%     if (M > current_num)            % A new change point has been detected
%         current_num= M;
%         plot_results(parameters, x, X, Y, Py, P, k);
%     end
% end

% Plot of entire data set:
chgpt_loc = plot_results(parameters, x, X, Y, Py, P, k);

% Eliminate unnecessary variables
clear i j J XTy beta_hat changepoint a back kk I num_chgpts r rr s_n v_n temp total_k v total new_x new_Y

