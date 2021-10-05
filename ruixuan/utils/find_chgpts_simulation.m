function chgpt_loc = find_chgpts_simulation(parameters, Py, P, k)
% This function draws samples from the exact posterior distribution on the
% number of change points and their locations after a new change point has 
% been detected within the simulation.  Function returns a vector 
% containing the number of times a change point was sampled at a particular 
% location.

N = length(Py);             % N = total # data points
%d_min = parameters(1);     % Unneeded parameters
%k_0 = parameters(2); 
%v_0 = parameters(3); sig_0 = parameters(4);
k_max = parameters(5);      % Maximum number of change points
num_samp = parameters(6);   % Number of sampled solutions


    % Variables used in sampling procedure:
    samp_holder=zeros(num_samp,k_max);  % Contains each of the num_samp change point solutions
    chgpt_loc=zeros(1,N);               % The number of times a change point is identified at each data point
    
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
                % TO AVOID UNDERFLOW, USE:
                % M_temp = max(temp); temp = temp - M_temp;
                total=log(sum(exp(temp)));
                temp(:)=exp(temp(:)-total);  % Normalize the vector
                changepoint=pick_k1(temp);   % Sample the location of the change point
                chgpt_loc(changepoint)= chgpt_loc(changepoint) +1; % Keep track of change point locations
                samp_holder(i,kk)=changepoint;  % Keep track of individual change point solutions
                
                             
                back=changepoint;   % Now work with the next segment
            end
            
            % The final changepoint
            %******** (ii) Sample the Location of the Change Points ***********
            kk=1;
            temp=zeros(1,back-1);
            for v=1:back-1
                temp(v)= Py(1,v)+Py(v+1,back); %Build the vector to sample from
            end
            % TO AVOID UNDERFLOW, USE:
            % M_temp = max(temp); temp = temp - M_temp;
            total=log(sum(exp(temp)));
            temp(:)=exp(temp(:)-total); % Normalize the vector
            changepoint=pick_k1(temp);  % Sample the location of the change point
            chgpt_loc(changepoint)= chgpt_loc(changepoint) +1; % Keep track of change point locations
            samp_holder(i,kk)=changepoint;  % Keep track of individual change point solutions
            
                    
        end
    end
    
end