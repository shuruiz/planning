function [locations, variances, found, detect] = process_chgpts(chgpt_loc, num_samp, CHGPTS, locations, variances, found, detect)
% Input: Set of sampled changepoint locations
% Output: Detection speed, bias, variance of posterior distribution, and 
% indication as to whether or not a change point has been 'detected' 

N = length(chgpt_loc);      % Current length of data set
C1 = CHGPTS(1); C2 = CHGPTS(2); C3 = CHGPTS(3); % Actual change point locations

start = C1-20;      % Set a boundary for what would qualify as 'detection'
if (N > start)      % If not, would indicate a false positive
    stop = min([N floor((C1+C2)/2) ]);  % Boundary for detection
    if(sum(chgpt_loc(start:stop)) > num_samp/2 && found(1) ==0)     
            %More than 50% of samples found a changepoint here and it has
            %not alread been detected
        found(1) = 1;      % First change point has been 'detected' 
        detect(1) = N-C1;  % Speed of detection
        temp = zeros(1,num_samp);
        counter = 1;       % Counts number of times change point was sampled within this region
        for i = start:stop
            if(chgpt_loc(i)>0)  % a change point was sampled at this location
                temp(counter:counter+chgpt_loc(i)-1) = i;
                counter = counter + chgpt_loc(i);
            end
        end
        temp = temp(1:counter-1);
        
        locations(1) = mean(temp)-C1;   
        % Difference between mean of posterior distribution and the actual location of the change point
        variances(1) = std(temp);
        % Standard deviation of the posterior distribution of the change point
    end
end


% Let's look for the second change point
start = ceil((C1+C2)/2);    % Set a boundary for what would qualify as 'detection'
if (N > start)
    stop = min([N floor((C2+C3)/2) ]);  % Boundary for detection
    if(sum(chgpt_loc(start:stop)) > num_samp/2 && found(2) ==0)     
            %More than 50% of samples found a changepoint here and it has
            %not alread been detected
        found(2) = 1;       % Second change point has been 'detected'
        detect(2) = N-C2;   % Speed of detection
        temp = zeros(1,num_samp);
        counter = 1;        % Counts number of times change point was sampled within this region
        for i = start:stop
            if(chgpt_loc(i)>0)  % a change point was sampled at this location
                temp(counter:counter+chgpt_loc(i)-1) = i;
                counter = counter + chgpt_loc(i);
            end
        end
        temp = temp(1:counter-1);
        
        locations(2) = mean(temp)-C2;
        % Difference between mean of posterior distribution and the actual location of the change point
        variances(2) = std(temp);
        % Standard deviation of the posterior distribution of the change point
    end
end

% And now the third change point
start = ceil((C2+C3)/2);    % Set a boundary for what would qualify as 'detection'
if (N > start)
    stop = min([N C3+20]);  % Boundary for detection
    if(sum(chgpt_loc(start:stop)) > num_samp/2 && found(3) ==0)     
            %More than 50% of samples found a changepoint here and it has
            %not alread been detected
        found(3) = 1;       % Third change point has been 'detected'
        detect(3) = N-C3;   % Speed of detection
        temp = zeros(1,num_samp);
        counter = 1;        % Counts number of times change point was sampled within this region
        for i = start:stop
            if(chgpt_loc(i)>0)  % a change point was sampled at this location
                temp(counter:counter+chgpt_loc(i)-1) = i;
                counter = counter + chgpt_loc(i);
            end
        end
        temp = temp(1:counter-1);
        
        locations(3) = mean(temp)-C3;
        % Difference between mean of posterior distribution and the actual location of the change point
        variances(3) = std(temp);
        % Standard deviation of the posterior distribution of the change point
    end
end


end