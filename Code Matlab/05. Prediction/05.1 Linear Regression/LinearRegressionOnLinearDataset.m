clc, clear, close all
%% Import Data
import = readmatrix('Car_dataset.txt');
import = import(:,1:8);

% Contents of Data
% #1 : mile per galon (fule efficiency)
% #2 : cylinders
% #3 : engine displacement in cubic inches
% #4 : horsepower
% #5 : weight
% #6 : acceleration
% #7 : year of model
% #8 : origin (1:America, 2:European, 3: Japanese)

%% Prepare data
% Target to predict is fule efficiency

data = import(:,2:8);
data(isnan(data)) = 0;
data = [ones(size(data,1),1),data];

target = import(:,1);

traindata = data(1:2:end,:);
traintarget = target(1:2:end,:);

% traindata = data(1:250,:);
% traintarget = target(1:250,:);

testdata = data(2:2:end,:);
testtarget = target(2:2:end,:);

% testdata = data(251:end,:);
% testtarget = target(251:end,:);


%% Pseudo inverse
[U,S,V] = svd(traindata,'econ');
m = V*inv(S)*U'*traintarget;

%% Prediction
% For validation first in train dataset

figure('Name','Predictions')
subplot(2,2,1)
hold on
plot(traintarget,'k-o')
plot(traindata*m,'r-o')
title('Prediction trainings dataset')
legend('True efficiency','Predicted efficiency')

[traintarget_sorted, ind] = sort(traintarget);
traindata_sorted = traindata(ind,:);

subplot(2,2,3)
hold on
plot(traintarget_sorted,'k-o')
plot(traindata_sorted*m,'r-o')
title('Prediction trainings dataset sorted')
legend('True efficiency','Predicted efficiency')

% Prediction on Testdata
subplot(2,2,2)
hold on
plot(testtarget,'k-o')
plot(testdata*m,'r-o')
title('Prediction test dataset')
legend('True efficiency','Predicted efficiency')

[testtarget_sorted, ind] = sort(testtarget);
testdata_sorted = testdata(ind,:);

subplot(2,2,4)
hold on
plot(testtarget_sorted,'k-o')
plot(testdata_sorted*m,'r-o')
title('Prediction test dataset sorted')
legend('True efficiency','Predicted efficiency')

%% Influence of the factors

data = data(:,2:8);
std_data = std(data,0);
mean_data = mean(data);

deviation_data = data-ones(size(data,1),1)*mean_data;

std_deviation_data = deviation_data./(ones(size(data,1),1)*std_data);

[U,S,V] = svd(std_deviation_data,'econ');
m = V*inv(S)*U'*target;

figure('Name','Influence of each factor in linear regression modle')
bar(m)
set(gca,'xticklabel',{'cylinders','engine displacement','weight',...
    'horsepower','acceleration','year of model','origin'})
title('Influence of individual factors on fuel efficiency (linear model)')
xlabel('Factors')
ylabel('Correlation')
