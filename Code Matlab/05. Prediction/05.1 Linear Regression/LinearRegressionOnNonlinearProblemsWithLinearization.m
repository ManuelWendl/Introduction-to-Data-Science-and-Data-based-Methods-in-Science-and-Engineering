clc, clear, close all
%% Load Data
import = readmatrix('standard-atmosphere.txt');
% contents:
% #1    altitude	
% #2    pressure	
% #3    temperature    
% #4    humidity
% 
% target is to predict the pressure from the 

data = import(:,[1,3,4]);
target = import(:,2);

[U,S,V] = svd(data,'econ');
m = V*inv(S)*U'*target;

figure('Name','Predicition')
hold on
plot(target,'k-o')
plot(data*m,'-or')
title('Non linearized Prediction')
xlabel('datasets')
ylabel('pressure')
legend('True Pressure','Predicted Pressure')

cap={'altitude','temperature','humidity'};

figure('Name','Correlations')
for i = 1:3
    subplot(1,3,i)
    plot(data(:,i),target,'o-')
    xlabel(cap{i});
    ylabel('pressure')
end

data = [log(data(:,1)+1),data(:,2:3)];

[U,S,V] = svd(data,'econ');
m = V*inv(S)*U'*target;

figure('Name','Predicition')
hold on
plot(target,'k-o')
plot(data*m,'-or')
title('Linearized Prediction')
xlabel('datasets')
ylabel('pressure')
legend('True Pressure','Predicted Pressure')
