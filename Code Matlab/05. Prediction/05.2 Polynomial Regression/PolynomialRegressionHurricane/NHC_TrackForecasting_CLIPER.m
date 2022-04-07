clc, clear
%% Import Preprocessed Data

Import = open('BestTrackProcessed2.mat');
data = Import.data;

% Description Data from Dataimport
%==========================================================================
%   #1    Calendar day 
%   #2    Latitude +24 
%   #3    Longitude +24
%   #4    Max sustained wind in knts +24
%   #5    Latitude +18 
%   #6    Longitude +18
%   #7    Max sustained wind in knts +18
%   #8    Latitude +12 
%   #9    Longitude +12
%   #10   Max sustained wind in knts +12
%   #11   Latitude +6 
%   #12   Longitude +6
%   #13   Max sustained wind in knts +6
%   #14   Latitude 0 
%   #15   Longitude 0
%   #16   Max sustained wind in knts 0
%   #17   Latitude -6 
%   #18   Longitude -6
%   #19   Max sustained wind in knts -6
%   #20   Latitude -12 
%   #21   Longitude -12
%   #22   Max sustained wind in knts -12
%   #23   Latitude -18 
%   #24   Longitude -18
%   #25   Max sustained wind in knts -18
%==========================================================================

%% Calculate factors according to Sivers 1998
% Parameters for Meridional CLIPER and target values
for i = 1:size(data,1)
    V0 = data(i,14)-data(i,17);
    V_12 = data(i,20)-data(i,23);
    V0V_12q = V0*(V_12)^2;
    W_71 = data(i,16)-71;
    W_71V_12 = W_71*V_12;
    W_71V0 = W_71*V0;
    U_12 = data(i,21)-data(i,24);
    V0qU_12 = V0^2*U_12;
    Y_24qV0 = (data(i,14)-24)^2*V0;
    D = data(i,1);
    D_248qV_12 = (D-248)^2*V_12;
    V0D_248q = V0*(D-248)^2;
    Y_24qD_248 = (data(i,14)-24)^2*(D-248);
    W_71D_248_V_12 = W_71*(D-248)*V_12;
    U0 = data(i,15)-data(i,18);
    D_248q = (D-248)^2;
    dataMeridional(i,:) = [data(i,14) ...
    V0, V_12, V0V_12q, W_71V_12, W_71V0, V0qU_12, Y_24qV0, D_248qV_12, ...
    V0D_248q, Y_24qD_248, W_71D_248_V_12, U0, D_248q];
    
    V_6 = data(i,11)-data(i,14);
    U_6 = data(i,12)-data(i,15);
    V_12 = data(i,8)-data(i,14);
    U_12 = data(i,9)-data(i,15);
    V_18 = data(i,5)-data(i,14);
    U_18 = data(i,6)-data(i,15);
    V_24 = data(i,2)-data(i,14);
    U_24 = data(i,3)-data(i,15);
    target(i,:) = [V_6,V_12,V_18,V_24,U_6,U_12,U_18,U_24];
end
% Original linear Regression form Neumann paper
    %dataMeridional = [ones(size(dataMeridional,1),1), dataMeridional];
% Optimized cubic regression     
    dataMeridional = [ones(size(dataMeridional,1),1),dataMeridional, dataMeridional.^2, dataMeridional.^3];

% Parameters for zonal prediction
for i = 1:size(data,1)
    U0 = data(i,15)-data(i,18);
    U_12 = data(i,21)-data(i,24);
    Y_24 = data(i,14)-24;
    V0 = data(i,14)-data(i,17);
    V0qU_12 = V0^2*U_12;
    Y0_24V0U_12 = (data(i,14)-24)*V0*U_12;
    X0_68 = data(i,15)-68;
    %additional ========
    W_12 = data(i,22)-data(i,25);
    %===================
    dataZonal(i,:) = [data(i,15), ...
    U0, U_12, Y_24, V0, V0qU_12, Y0_24V0U_12, X0_68, W_12];    
end
% Original linear Regression form Neumann paper
    %dataZonal = [ones(size(dataZonal,1),1), dataZonal];
% Optimized cubic regression  
    dataZonal = [ones(size(dataZonal,1),1), dataZonal, dataZonal.^2, dataZonal.^3];

r = 32000;
trainZonal = dataZonal(1:r,:);
trainMeridional = dataMeridional(1:r,:);
trainTarget = target(1:r,:);

testZonal = dataZonal(r:end,:);
testMeridional = dataMeridional(r:end,:);
testTarget = target(r:end,:);

%% SVD for linear regression

[Uz,Sz,Vz] = svd(trainZonal,'econ');
[Um,Sm,Vm] = svd(trainMeridional,'econ');

%% Determine Regression Constants

%xz6 = Vz*inv(Sz)*Uz'*trainTarget(:,5);
xz6 = pinv(trainZonal)*trainTarget(:,5);
%xz12 = Vz*inv(Sz)*Uz'*trainTarget(:,6);
xz12 = pinv(trainZonal)*trainTarget(:,6);
%xz18 = Vz*inv(Sz)*Uz'*trainTarget(:,7);
xz18 = pinv(trainZonal)*trainTarget(:,7);
%xz24 = Vz*inv(Sz)*Uz'*trainTarget(:,8);
xz24 = pinv(trainZonal)*trainTarget(:,8);

xm6 = Vm*inv(Sm)*Um'*trainTarget(:,1);
xm12 = Vm*inv(Sm)*Um'*trainTarget(:,2);
xm18 = Vm*inv(Sm)*Um'*trainTarget(:,3);
xm24 = Vm*inv(Sm)*Um'*trainTarget(:,4);

%% Test Regression Modle on Test Set

figure('Name','Prediction test on test data')
subplot(2,4,1)
hold on
plot(testTarget(1:500,5),'k-o')
plot(testZonal(1:500,:)*xz6,'r-o')
title('Zonal Prediction +6hrs')
legend('True Data','Prediction')

subplot(2,4,2)
hold on
plot(testTarget(1:500,6),'k-o')
plot(testZonal(1:500,:)*xz12,'r-o')
title('Zonal Prediction +12hrs')
legend('True Data','Prediction')

subplot(2,4,3)
hold on
plot(testTarget(1:500,7),'k-o')
plot(testZonal(1:500,:)*xz18,'r-o')
title('Zonal Prediction +18hrs')
legend('True Data','Prediction')

subplot(2,4,4)
hold on
plot(testTarget(1:500,8),'k-o')
plot(testZonal(1:500,:)*xz24,'r-o')
title('Zonal Prediction +24hrs')
legend('True Data','Prediction')

subplot(2,4,5)
hold on
plot(testTarget(1:500,1),'k-o')
plot(testMeridional(1:500,:)*xm6,'r-o')
title('Meridional Prediction +6hrs')
legend('True Data','Prediction')

subplot(2,4,6)
hold on
plot(testTarget(1:500,2),'k-o')
plot(testMeridional(1:500,:)*xm12,'r-o')
title('Meridional Prediction +12hrs')
legend('True Data','Prediction')

subplot(2,4,7)
hold on
plot(testTarget(1:500,3),'k-o')
plot(testMeridional(1:500,:)*xm18,'r-o')
title('Meridional Prediction +18hrs')
legend('True Data','Prediction')

subplot(2,4,8)
hold on
plot(testTarget(1:500,4),'k-o')
plot(testMeridional(1:500,:)*xm24,'r-o')
title('Meridional Prediction +24hrs')
legend('True Data','Prediction')

% Sort Plot
[testTargetsortz6, indz6] = sort(testTarget(:,5));
[testTargetsortz12, indz12] = sort(testTarget(:,6));
[testTargetsortz18, indz18] = sort(testTarget(:,7));
[testTargetsortz24, indz24] = sort(testTarget(:,8));
[testTargetsortm6, indm6] = sort(testTarget(:,1));
[testTargetsortm12, indm12] = sort(testTarget(:,2));
[testTargetsortm18, indm18] = sort(testTarget(:,3));
[testTargetsortm24, indm24] = sort(testTarget(:,4));

figure('Name','Prediction test on test data')
subplot(2,4,1)
hold on
plot(testZonal(indz6,:)*xz6,'r-')
plot(testTargetsortz6,'k-')
title('Zonal Prediction +6hrs')
legend('Prediction','True Data')

subplot(2,4,2)
hold on
plot(testZonal(indz12,:)*xz12,'r-')
plot(testTargetsortz12,'k-')
title('Zonal Prediction +12hrs')
legend('Prediction','True Data')

subplot(2,4,3)
hold on
plot(testZonal(indz18,:)*xz18,'r-')
plot(testTargetsortz18,'k-')
title('Zonal Prediction +18hrs')
legend('Prediction','True Data')

subplot(2,4,4)
hold on
plot(testZonal(indz24,:)*xz24,'r-')
plot(testTargetsortz24,'k-')
title('Zonal Prediction +24hrs')
legend('Prediction','True Data')

subplot(2,4,5)
hold on
plot(testMeridional(indm6,:)*xm6,'r-')
plot(testTargetsortm6,'k-')
title('Meridional Prediction +6hrs')
legend('Prediction','True Data')

subplot(2,4,6)
hold on
plot(testMeridional(indm12,:)*xm12,'r-')
plot(testTargetsortm12,'k-')
title('Meridional Prediction +12hrs')
legend('Prediction','True Data')

subplot(2,4,7)
hold on
plot(testMeridional(indm18,:)*xm18,'r-')
plot(testTargetsortm18,'k-')
title('Meridional Prediction +18hrs')
legend('Prediction','True Data')

subplot(2,4,8)
hold on
plot(testMeridional(indm24,:)*xm24,'r-')
plot(testTargetsortm24,'k-')
title('Meridional Prediction +24hrs')
legend('Prediction','True Data')

% Error plot

errorz6 = sum(abs(testTargetsortz6-testZonal(indz6,:)*xz6))/length(testTargetsortz6);
errorz12 = sum(abs(testTargetsortz12-testZonal(indz12,:)*xz12))/length(testTargetsortz12);
errorz18 = sum(abs(testTargetsortz18-testZonal(indz18,:)*xz18))/length(testTargetsortz18);
errorz24 = sum(abs(testTargetsortz24-testZonal(indz24,:)*xz24))/length(testTargetsortz24);
errorz = [errorz6, errorz12, errorz18, errorz24];
errorm6 = sum(abs(testTargetsortm6-testMeridional(indm6,:)*xm6))/length(testTargetsortm6);
errorm12 = sum(abs(testTargetsortm12-testMeridional(indm12,:)*xm12))/length(testTargetsortm12);
errorm18 = sum(abs(testTargetsortm18-testMeridional(indm18,:)*xm18))/length(testTargetsortm18);
errorm24 = sum(abs(testTargetsortm24-testMeridional(indm24,:)*xm24))/length(testTargetsortm24);
errorm = [errorm6, errorm12, errorm18, errorm24];
error = sqrt(errorm.^2+errorz.^2);
figure('Name','Error')
hold on
plot([6, 12, 18, 24],error*60,'-o')
plot([6, 12, 18, 24],errorz*60,'-o')
plot([6, 12, 18, 24],errorm*60,'-o')
legend( 'Total Error Prediction +6 +12 +18 +24','Error Zonal Prediction +6 +12 +18 +24','Error Meridional Prediction +6 +12 +18 +24')
ylabel('Error in mi')

%% Visualation of Forecast

%Pick arbitrary Hurricans
%HCnr = randi(size(testTarget,1),1,15);
HCnr = [5077,5200,5782,3457,5232,4532,277,934,1493,1579,1090,1504,1763,6088,2218];

figure('Name','Track Vizualisation')
image('CData',imread('Map.jpg'),'XData',[-7*180/12, 180/12+0.5],'YData',[4*90/6, -90/6])
hold on
axis([-7*180/12, -180/12, 90/12, 4*90/6])
xlabel('Longitude in degrees')
ylabel('Latitude in degrees')
for i = 1:length(HCnr)
    HRz = testZonal(HCnr(i),:);
    HRm = testMeridional(HCnr(i),:);
    
    Track = testTarget(HCnr(i),:);
    Track(1:4)=Track(1:4)+HRm(2);
    Track(5:8)=Track(5:8)+HRz(2);
    
    Track = [HRm(2) Track(1:4) HRz(2) Track(5:8)];
    Trackpred = [HRm(2), HRm*xm6+HRm(2), HRm*xm12+HRm(2), ...
    HRm*xm18+HRm(2), HRm*xm24+HRm(2), ...
    HRz(2), HRz*xz6+HRz(2), HRz*xz12+HRz(2), HRz*xz18+HRz(2),...
    HRz*xz24+HRz(2),]; 
    plot(-Track(6:10),Track(1:5),'-xk','LineWidth',1.5)
    plot(-Trackpred(6:10),Trackpred(1:5),'-dr','LineWidth',1.5)
    leg = legend('True Track','Predicted Track');
    title(leg, 'Track Positions: 0,+6,+12,+18,+24')
end

