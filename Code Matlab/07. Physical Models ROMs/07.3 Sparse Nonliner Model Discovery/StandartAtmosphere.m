clc, clear, close all
%% Load Dataset: 

% This problem has the following inputs: 
% 1. Frequency, in Hertzs. 
% 2. Angle of attack, in degrees. 
% 3. Chord length, in meters. 
% 4. Free-stream velocity, in meters per second. 
% 5. Suction side displacement thickness, in meters. 
% 
% The target is: 
% 6. Scaled sound pressure level, in decibels. 


import = importdata('AirfoilSelfNoise.csv');
data = import.data(:,1:5);
target = import.data(:,6);
labels = import.colheaders(1:5);
labels = regexp(labels,'[^""]*','match','once');

%% Preprocess Data
% Decibels is 10*log_10 of two quantities. 

%target = 10.^(1/10*target)/1e11;
data(:,1) = data(:,1)/1000;

% Split Data
testdata = data(1:188,:);
data = data(189:end,:);

testtarget = target(1:188);
target = target(189:end);

%% Build non Linear Library
[Phi, Philabels] = buildNonLinearLibrary(data,labels); 
[testPhi, ~] = buildNonLinearLibrary(testdata,labels); 

%% Sparse Linear Solver (For different lambda and trade off curve)
e = logspace(-2,2,10);
acitveTerms = zeros(size(e));
l2norm = zeros(size(e));
Eta = zeros(size(Phi,2),length(e));

for i = 1:length(e)
    cvx_begin;
     cvx_solver mosek
         variable eta(size(Phi,2)); % sparse vector of coefficients 
         minimize(norm(Phi*eta - target,2) + e(i)*norm(eta,1)) 
    cvx_end;

    % Truncation and Active Terms in model
    activeTerms(i) = sum(abs(eta)>1e-5);
    eta(abs(eta)<1e-5) = 0;
    % L2 error
    l2norm(i) = norm(Phi*eta - target,2)/size(Phi,1);
    Eta(:,i) = eta;
end

figure('Name','TradeOff')
plot(activeTerms,l2norm)
xlabel('active terms'),ylabel('||Phi*eta-target||_2 / datapoint')

figure('Name','Training Data')
for i = 1:length(e)
    subplot(2,5,i)
    plot(Phi*Eta(:,i),'r-o'); hold on
    plot(target,'k--d');
    title(['active terms = ',num2str(activeTerms(i))])
end

figure('Name','Test Data')
for i = 1:length(e)
    subplot(2,5,i)
    plot(testPhi*Eta(:,i),'r-o'); hold on
    plot(testtarget,'k--d');
    title(['active terms = ',num2str(activeTerms(i))])
end


%% Function to Build Non-Linear Library
function [Phi, labels] = buildNonLinearLibrary(inputData,inputLabels)
    % Initialise 
    Phi = [];
    labels = {};

    % Modify inputData
    [predictors,axis] = min(size(inputData));
    if axis == 1
        inputData = inputData';
    end

    % Add unit vector to Library
    Phi(:,end+1) = ones(size(inputData,1),1);
    labels{end+1} = 'Unity';

    for i=1:predictors
        Phi(:,end+1) = inputData(:,i);
        labels{end+1} = inputLabels(i);

        Phi(:,end+1) = sin(inputData(:,i));
        labels{end+1} = strcat('sin(',inputLabels(i),')');

        Phi(:,end+1) = cos(inputData(:,i));
        labels{end+1} = strcat('cos(',inputLabels(i),')');

        Phi(:,end+1) = tan(inputData(:,i));
        labels{end+1} = strcat('tan(',inputLabels(i),')');

        for ii=1:predictors
            Phi(:,end+1) = inputData(:,i).*inputData(:,ii);
            labels{end+1} = strcat(inputLabels(i),'*',inputLabels(ii));

            Phi(:,end+1) = sin(inputData(:,i)).*inputData(:,ii);
            labels{end+1} = strcat('sin(',inputLabels(i),')','*',inputLabels(ii));

            Phi(:,end+1) = sin(inputData(:,i).*inputData(:,ii));
            labels{end+1} = strcat('sin(',inputLabels(i),'*',inputLabels(ii),')');

            Phi(:,end+1) = cos(inputData(:,i)).*inputData(:,ii);
            labels{end+1} = strcat('cos(',inputLabels(i),')*',inputLabels(ii));

            Phi(:,end+1) = cos(inputData(:,i).*inputData(:,ii));
            labels{end+1} = strcat('cos(',inputLabels(i),'*',inputLabels(ii),')');

            Phi(:,end+1) = tan(inputData(:,i)).*inputData(:,ii);
            labels{end+1} = strcat('tan(',inputLabels(i),')*',inputLabels(ii));

            Phi(:,end+1) = tan(inputData(:,i).*inputData(:,ii));
            labels{end+1} = strcat('tan(',inputLabels(i),'*',inputLabels(ii),')');

            if sum(inputData(:,ii) == 0,'all') == 0
                Phi(:,end+1) = inputData(:,i)./inputData(:,ii);
                labels{end+1} = strcat(inputLabels(i),'/',inputLabels(ii));

                Phi(:,end+1) = sin(inputData(:,i))./inputData(:,ii);
                labels{end+1} = strcat('sin(',inputLabels(i),')/',inputLabels(ii));
    
                Phi(:,end+1) = sin(inputData(:,i)./inputData(:,ii));
                labels{end+1} = strcat('sin(',inputLabels(i),'/',inputLabels(ii),')');
    
                Phi(:,end+1) = cos(inputData(:,i))./inputData(:,ii);
                labels{end+1} = strcat('cos(',inputLabels(i),')/',inputLabels(ii));
    
                Phi(:,end+1) = cos(inputData(:,i)./inputData(:,ii));
                labels{end+1} = strcat('cos(',inputLabels(i),'/',inputLabels(ii),')');

                Phi(:,end+1) = tan(inputData(:,i))./inputData(:,ii);
                labels{end+1} = strcat('tan(',inputLabels(i),')/',inputLabels(ii));
    
                Phi(:,end+1) = tan(inputData(:,i)./inputData(:,ii));
                labels{end+1} = strcat('tan(',inputLabels(i),'/',inputLabels(ii),')');
            end

            for iii=1:predictors
                Phi(:,end+1) = inputData(:,i).*inputData(:,ii).*inputData(:,iii);
                labels{end+1} = strcat(inputLabels(i),'*',inputLabels(ii),'*',inputLabels(iii));

                Phi(:,end+1) = sin(inputData(:,i)).*inputData(:,ii).*inputData(:,iii);
                labels{end+1} = strcat('sin(',inputLabels(i),')*',inputLabels(2),'*',inputLabels(iii));
               
                Phi(:,end+1) = sin(inputData(:,i).*inputData(:,ii)).*inputData(:,iii);
                labels{end+1} = strcat('sin(',inputLabels(i),'*',inputLabels(2),')*',inputLabels(iii));

                Phi(:,end+1) = sin(inputData(:,i).*inputData(:,ii).*inputData(:,iii));
                labels{end+1} = strcat('sin(',inputLabels(i),'*',inputLabels(2),'*',inputLabels(iii),')');

                Phi(:,end+1) = cos(inputData(:,i)).*inputData(:,ii).*inputData(:,iii);
                labels{end+1} = strcat('cos(',inputLabels(i),')*',inputLabels(2),'*',inputLabels(iii));
               
                Phi(:,end+1) = cos(inputData(:,i).*inputData(:,ii)).*inputData(:,iii);
                labels{end+1} = strcat('cos(',inputLabels(i),'*',inputLabels(2),')*',inputLabels(iii));

                Phi(:,end+1) = cos(inputData(:,i).*inputData(:,ii).*inputData(:,iii));
                labels{end+1} = strcat('cos(',inputLabels(i),'*',inputLabels(2),'*',inputLabels(iii),')');

                Phi(:,end+1) = tan(inputData(:,i)).*inputData(:,ii).*inputData(:,iii);
                labels{end+1} = strcat('tan(',inputLabels(i),')*',inputLabels(2),'*',inputLabels(iii));
               
                Phi(:,end+1) = tan(inputData(:,i).*inputData(:,ii)).*inputData(:,iii);
                labels{end+1} = strcat('tan(',inputLabels(i),'*',inputLabels(2),')*',inputLabels(iii));

                Phi(:,end+1) = tan(inputData(:,i).*inputData(:,ii).*inputData(:,iii));
                labels{end+1} = strcat('tan(',inputLabels(i),'*',inputLabels(2),'*',inputLabels(iii),')');

                if sum(inputData(:,iii) == 0,'all') == 0
                    Phi(:,end+1) = (inputData(:,i).*inputData(:,ii))./inputData(:,iii);
                    labels{end+1} = strcat('(',inputLabels(i),'*',inputLabels(ii),')/',inputLabels(iii));
                end
                if sum(inputData(:,ii).*inputData(:,iii) == 0,'all') == 0
                    Phi(:,end+1) = inputData(:,i)./(inputData(:,ii).*inputData(:,iii));
                    labels{end+1} = strcat(inputLabels(i),'/(',inputLabels(ii),'*',inputLabels(iii),')');                
                end
            end
        end
    end
end