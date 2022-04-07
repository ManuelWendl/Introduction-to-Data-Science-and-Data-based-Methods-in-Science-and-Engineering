clc, clear, close all
%% Import Data
Imp = importdata('yacht_hydrodynamics.dat');

% Contents:
% 1. Longitudinal position of the center of buoyancy, adimensional. 
% 2. Prismatic coefficient, adimensional. 
% 3. Length-displacement ratio, adimensional. 
% 4. Beam-draught ratio, adimensional. 
% 5. Length-beam ratio, adimensional. 
% 6. Froude number, adimensional. 
% The measured variable is the residuary resistance per unit weight of displacement: 
% 7. Residuary resistance per unit weight of displacement, adimensional. 
%% Prepare  Data
data = [ones(size(Imp,1),1),Imp(:,1:6)];
target =  Imp(:,7);

% Predicting data for new airfoil length

traindata =  data(1:196,:);
testdata =  data(197:end,:);

traintarget = target(1:196,:);
testtarget = target(197:end,:);

%% Regression Model linear
[U,S,V] = svd(traindata,'econ');
c = V*inv(S)*U'*traintarget;

figure('Name','Prediction')
subplot(2,3,1)
hold on
plot(testtarget,'k-o');
plot(testdata*c,'r-o');
title('Linear regression model')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')


[testtarget_s, ind] = sort(testtarget);
testdata_s = testdata(ind,:);

subplot(2,3,4)
hold on
plot(testtarget_s,'k-o');
plot(testdata_s*c,'r-o');
title('Linear regression model sorted')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')


%% Quadratic Regression
data = [data, Imp(:,1:6).^2];

traindata =  data(1:196,:);
testdata =  data(197:end,:);

c = pinv(traindata)*traintarget;


subplot(2,3,2)
hold on
plot(testtarget,'k-o');
plot(testdata*c,'r-o');
title('Quadratic regression model')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')

testdata_s = testdata(ind,:);

subplot(2,3,5)
hold on
plot(testtarget_s,'k-o');
plot(testdata_s*c,'r-o');
title('Quadratic regression model sorted')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')

%% Cubic Regression

data = [data, Imp(:,1:6).^2, Imp(:,1:6).^3];

traindata =  data(1:196,:);
testdata =  data(197:end,:);

c = pinv(traindata)*traintarget;

subplot(2,3,3)
hold on
plot(testtarget,'k-o');
plot(testdata*c,'r-o');
title('Cubic regression model')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')

testdata_s = testdata(ind,:);

subplot(2,3,6)
hold on
plot(testtarget_s,'k-o');
plot(testdata_s*c,'r-o');
title('Cubic regression model sorted')
legend('True residuary resistance','Predicted residuary resistance',...
    'Location','northwest')