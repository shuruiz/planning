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

%********** SIMULATION OF A DATA SET WITH THREE CHANGE POINTS *************
%********************** Section 5.2 of Article ****************************

NOISE = sqrt(1);    % Magnitude of added noise
num_data = 175;     % Number of observations in each data set
num_sim = 1000;     % Number of simulations to generate
found_mode = zeros(num_sim,3); found_median = zeros(num_sim,3); found_mean = zeros(num_sim,3); found_total = zeros(num_sim,3);
% Indicates whether or not a particular change point has been detected for
% each of the three detection criteria in a given simulation
locations_mode = zeros(num_sim,3); locations_median = zeros(num_sim,3); locations_mean = zeros(num_sim,3); locations_total = zeros(num_sim,3);
% The difference between the inferred and true change point location for 
% each of the three change points in each simulation. 
variances_mode = zeros(num_sim,3); variances_median = zeros(num_sim,3); variances_mean = zeros(num_sim,3); variances_total = zeros(num_sim,3);
% Standard deviation of the posterior distribution for each of the change
% points identified by the three detection criteria.
detect_mode = zeros(num_sim,3); detect_median = zeros(num_sim,3); detect_mean = zeros(num_sim,3);
% Specifies the detection speed for each of the three detection criteria
% for each simulation.  Matrix entry is the number of data points after the
% true change point location at which detection occurred.

for SIM = 1:num_sim

clear x_complete Y_complete
%************(1)Load the Data*****************************

x_complete=(1:num_data)';    % Time points


sign = 2*round(rand(4,1))-1;    % Negates the change in trend with probability 0.5 (see below)

T1 = 0.15 + 0.05*randn();    T1 = sign(1)*T1;
        % These represent the change in the trend, rather than its actual magnitude
T2 = 0.4 + 0.05*randn();    T2 = T1 + sign(2)*T2;
T3 = 0.4 + 0.025*randn();  T3 = T2 + sign(3)*T3;
T4 = 0.25 + 0.01*randn();  T4 = T3 + sign(4)*T4;
intercept = -2 + 4*rand();

C1 = 65 + floor(11*rand());     %Change point locations
C2 = 85 + floor(11*rand());
C3 = 120 + floor(11*rand());
CHGPTS = [C1 C2 C3];

Y_complete(1:C1) = intercept + T1*(1:C1)' + NOISE*randn(C1,1);
Y_complete(C1+1:C2) = Y_complete(C1) + T2*(1:C2-C1)' + NOISE*randn(C2-C1,1);
Y_complete(C2+1:C3) = Y_complete(C2) + T3*(1:C3-C2)' + NOISE*randn(C3-C2,1);
Y_complete(C3+1:num_data) = Y_complete(C3) + T4*(1:num_data-C3)' + NOISE*randn(num_data-C3,1);
Y_complete = Y_complete';


%*************(2)Define the Parameter Values**************

N=5;            % Number of data points in initial data set.  
                % If you want to look at entire data set at once, 
                % set N = length(Y_complete)
k_max=6;        % Maximum number of change points
d_min=5;        % Minimum distance between adjacent change points
      
k_0=0.1;           % Hyperparameter for the prior on the regression coefficients
v_0=1; sig_0=1;     % Hyperparameter for scaled inverse chi-square prior on the variance

Y = Y_complete(1:N);        % Work with initial data set only
x = x_complete(1:N);
X = [zeros(N,1)+1 (1:N)'];  % Build the regression model - in this case, a linear model
                            % Column 1 represents the constant term
                            % Column 2 represents the linear trend

[~, m] =size(X);            % r = total # data points (N), m = # regressors
beta0= zeros(m,1);          % Mean of multivariate normal prior on regression coefficients
num_samp=500;               % Number of sampled solutions

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
            - log(gamma(v_0/2)) - v_n*log(s_n/2)/2 - a*log(2*pi)/2 - log(det(J))/2 ;    %Minus because J is no longer inverted!
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
k(:)=exp(k(:)-total_k);         % Normalize the vector - Equation (5) and (10)

%If analyzing the complete data set all at once you may want to use the
%next line of code:
% plot_results(parameters, x, X, Y, Py, P, k);

%********************ADDING A NEW DATA POINT ***************************
% This part of the code loops through all remaining observations in the
% data set.  If you are adding only a single observation, then you can
% comment out the loop

current_num1=zeros(1,num_data); current_num2=zeros(1,num_data); current_num3=zeros(1,num_data);
% Assumes no change points have yet to be detected for any of three detection criteria
KK=zeros(k_max+1,N);    % A record of the posterior distribution of k as new observations are added


for n = N+1:length(Y_complete)
    % Define variables: new_x [Time Point], new_Y [Data Point] and append these
    % to the existing x, and Y.  Also, update N and the regression model, X.
    % Example - Taking the next data point in the simulated data set
    new_Y = Y_complete(n);      Y = [Y; new_Y];
    new_x = x_complete(n);      x = [x; new_x];
    N = N+1;
    X = [X; 1 N];       % Linear Model
    
    %Function call to update Py and P matrices:
    [Py, P] = addone(parameters, X, Y, Py, P);
    
    
% From here, you'll have to decide whether or not to place a new change
% point.  One way could be to look at how the distribution on k (the posterior
% distribution on the # change points) changes as you add more data points.

    k= P(:,N);
    for i=1:k_max                   % Equation (3)
        if(N-(i+1)*d_min+i >= i)    % If a plausible solution exists...
            k(i) = k(i) + log(0.5) - log(k_max) - log(nCk(N-(i+1)*d_min+i,i));
                                                    % Above term is N_k
        end
    end
    k=[Py(1,N)+ log(0.5); k];       % Zero change points
    
    total_k = log(sum(exp(k)));     % Adds logarithms and puts answer in log form
    k(:)=exp(k(:)-total_k);         % Normalize the vector - Equations (4) and (7)
    
    
    %**** Detecting Change Points *******
    
    KK = [KK k];
    
    
    %Use the MODE
    [~, M1] = max(k); M1 = M1-1;        %adjusts for 0 change points
    current_num1(n) = M1;
    
    
    %Use the MEDIAN
    temp=0; M2 = 0;
    while(temp<0.5)
    M2=M2+1;
    temp = temp+k(M2);
    end
    M2=M2-1;                          %adjusts for 0 change points
    current_num2(n) = M2;
    
    
    %Use the MEAN
    temp=0;
    for i = 1:length(k)
        temp = temp+(i-1)*k(i);
    end
    M3= round(temp);
    current_num3(n) = M3;
    
    if (M1 > max(current_num1(n-d_min:n-1)))    
        % If the current number of change points is larger than the number
        % of change points recently detected...
        % (this check is necessary because the posterior distribution is
        % not a monotone function)
        chgpt_loc = find_chgpts_simulation(parameters, Py, P, k);
        % Simulates from the posterior distribution
        [locations_mode(SIM,:), variances_mode(SIM,:), found_mode(SIM,:), detect_mode(SIM,:)] = process_chgpts(chgpt_loc, num_samp, CHGPTS, locations_mode(SIM,:), variances_mode(SIM,:), found_mode(SIM,:), detect_mode(SIM,:));
        % Processes the above chgpt_loc vector to determine which change
        % points have been 'detected', how fast, and the mean and standard
        % deviation of the posterior distribution of the change point
        % location
    end
    
    if (M2 > max(current_num2(n-d_min:n-1)))
        chgpt_loc = find_chgpts_simulation(parameters, Py, P, k);
        [locations_median(SIM,:), variances_median(SIM,:), found_median(SIM,:), detect_median(SIM,:)] = process_chgpts(chgpt_loc, num_samp, CHGPTS, locations_median(SIM,:), variances_median(SIM,:), found_median(SIM,:), detect_median(SIM,:));
    end
    
    
    if (M3 > max(current_num3(n-d_min:n-1)))
        chgpt_loc = find_chgpts_simulation(parameters, Py, P, k);
        [locations_mean(SIM,:), variances_mean(SIM,:), found_mean(SIM,:), detect_mean(SIM,:)] = process_chgpts(chgpt_loc, num_samp, CHGPTS, locations_mean(SIM,:), variances_mean(SIM,:), found_mean(SIM,:), detect_mean(SIM,:));
    end
    
end

% How do the results change if we had the entire data set?  Batch Analysis!
chgpt_loc = find_chgpts_simulation(parameters, Py, P, k);
[locations_total(SIM,:), variances_total(SIM,:), found_total(SIM,:), ~] = process_chgpts(chgpt_loc, num_samp, CHGPTS, zeros(1,3), zeros(1,3), zeros(1,3), zeros(1,3));
        
% NOTE:  If at any point you would like to visualize a particular
% simulation, you can use the command:
% plot_results(parameters, x, X, Y, Py, P, k);
% Just adjust the axis boundaries within this function accordingly

end

disp('Average locations: Mean, Median, Mode, Total')
locations = zeros(4,3);
for i = 1:3
    disp(['changepoint: ' num2str(i)])
    locations(1,i) =sum(locations_mean(:,i))/sum(found_mean(:,i));
    locations(2,i) =sum(locations_median(:,i))/sum(found_median(:,i));
    locations(3,i) =sum(locations_mode(:,i))/sum(found_mode(:,i));
    locations(4,i) =sum(locations_total(:,i))/sum(found_total(:,i));
end
disp(locations)

disp('Standard Deviation of Location: Mean, Median, Mode, Total');
%std(locations_median(found_median(:,3)~=0, 3))
stdev_loc = zeros(4,3);
for i = 1:3
    disp(['changepoint: ' num2str(i)])
    stdev_loc(1,i) =std(locations_mean(found_mean(:,i)~=0, i));
    stdev_loc(2,i) =std(locations_median(found_median(:,i)~=0, i));
    stdev_loc(3,i) =std(locations_mode(found_mode(:,i)~=0, i));
    stdev_loc(4,i) =std(locations_total(found_total(:,i)~=0, i));
end
disp(stdev_loc)

disp('Average Standard Deviation: Mean, Median, Mode, Total')
variances = zeros(4,3);
for i = 1:3
    disp(['changepoint: ' num2str(i)])
    variances(1,i) =sum(variances_mean(:,i))/sum(found_mean(:,i));
    variances(2,i) =sum(variances_median(:,i))/sum(found_median(:,i));
    variances(3,i) =sum(variances_mode(:,i))/sum(found_mode(:,i));
    variances(4,i) =sum(variances_total(:,i))/sum(found_total(:,i));
end
disp(variances)

disp('Average Detection Speed: Mean, Median, Mode')
detect = zeros(3,3);
for i = 1:3
    disp(['changepoint: ' num2str(i)])
    detect(1,i) = sum(detect_mean(:,i))/sum(found_mean(:,i));
    detect(2,i) = sum(detect_median(:,i))/sum(found_median(:,i));
    detect(3,i) = sum(detect_mode(:,i))/sum(found_mode(:,i));
end
disp(detect)

disp('Detection Rate: Mean, Median, Mode, Total')
pct_mean = sum(found_mean)/num_sim;
pct_median = sum(found_median)/num_sim;
pct_mode = sum(found_mode)/num_sim;
pct_total = sum(found_total)/num_sim;
disp(pct_mean); disp(pct_median); disp(pct_mode); disp(pct_total);

% Eliminate unnecessary variables
clear i j J XTy beta_hat changepoint a back kk num_chgpts r rr s_n v_n temp total_k v total new_x new_Y
clear I SIM m n sign x Y C1 C2 C3 CHGPTS M1 M2 M3 KK T1 T2 T3 T4 intercept
clear x_complete Y_complete beta0 N P Py X
clear chgpt_loc current_num1 current_num2 current_num3
clear NOISE d_min k k_0 k_max num_data num_samp num_sim sig_0 v_0 parameters


