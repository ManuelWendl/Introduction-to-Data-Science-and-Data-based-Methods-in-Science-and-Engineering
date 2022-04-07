clc, clear, close all
%% Definition of Referance and Disturbance
t = 0:.01:10;
wr = 60*ones(size(t));      % reference speed
disturbance = 10*sin(pi*t)-20;

%% Definition of Model;
Model = 1; % y = u input equal to output

%% Open Loop response
yOL = wr*Model + disturbance;

%% Closed Loop response
K = 50;
yCl = wr*(2*K)/(1+2*K);

figure(1)
hold on
plot(t,wr); plot(t,disturbance,'k--'); plot(t,yOL); plot(t,yCl); 
legend('Reference Speed','Disturbance','Open Loop Controller','Closed Loop Controller');
xlabel('t'); ylabel('v'); xlim([0,10]); ylim([-40,70]);