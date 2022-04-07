%**************************************************************************
%   Covid19 Simulation Model_2
%   ******************
%   Modelled Groups:
%   S:  Susceptibel Population (Unprotected Population)
%   I:  Infected Population responsible for spreading (Currently spreading the virus)
%   P:  Protected Population (Recovered from Virus w.o. vaccination)
%   Id: Identified Infected Population (Positively tested) 
%   E:  Exposed Population (Got Infected and are in Incubation Period)
%   D:  Dead Population
%   R:  Recovered Population
%
%   Modelling Parameters:
%   k:  Transition Rate S->I
%   tr: Average recovery time 
%   kps: Transition Rate P->S 
%   ti: Incubation Period
%   tid: Time until positive tested
%   d:  Fraction of detected
%   l:  Mortality
%   td: Average Death Time
%   v:  Daily Vaccination Rate
%   a:  Protection Factor (In Presence of Pandemic the awareness rises)
%   krp: Characteristic reinfection rate
%**************************************************************************
clc, clear, close all
%% Initial Conditions
t0 = 56; tn = 480; dt = 1/8; 
S0 = 10.2e6; I0 = 3; P0 = 0; Id0 = 0; E0 = 0; D0 = 0; R0 = 0;
y0 = [P0,S0,E0,I0,Id0,R0,D0,0,0];

% Constant Parameters
a=1.3e-3; ti=3.5; tid=6.35; d=0.25; l=0.02; tr=14; krp=1e-2; v=0;
% Simulation first time period
options = odeset('NonNegative',[1,2,3,4,5,6,7]);
tspan = [56,91];
kps=0; k=0.6; td=5.5;
[t1,y1] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation second time period
tspan = [92,132]; y0 = y1(end,:);
kps=2.7e-2; k=0.9; td=5.5;
[t2,y2] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation third time period
tspan = [133,244]; y0 = y2(end,:);
kps=5.5e-2; k=0.9; td=16;
[t3,y3] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation fourth time period
tspan = [245,279]; y0 = y3(end,:);
kps=8.3e-2; k=0.3; td=16;
[t4,y4] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation fifth time period
tspan = [280,344]; y0 = y4(end,:);
kps=1.6; k=0.3; td=16;
[t5,y5] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation sixth time period
tspan = [345,359]; y0 = y5(end,:);
kps=1.6; k=0.3; td=9.6;
[t6,y6] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation seventh time period
tspan = [360,384]; y0 = y6(end,:);
kps=27.8; k=0.3; td=9.6;
[t7,y7] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Simulation eighth time period
tspan = [385,420]; y0 = y7(end,:);
kps=1.8e-2; k=0.3; td=50;
[t8,y8] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);

t = [t1;t2;t3;t4;t5;t6;t7;t8];
y = [y1;y2;y3;y4;y5;y6;y7;y8];

figure(1)
subplot(2,2,1)
plot(t,y(:,4:5)); legend('Infected','Identified Infected'); xlabel('d'); ylabel('cases');
title('Infected and Identified Infected'); xlim([56,420]);
subplot(2,2,2)
semilogy(t,y(:,9)); xlabel('d'); ylabel('cases')
title('Total Detected Cases'); xlim([56,420]); ylim([1e0,1e7]);
subplot(2,2,3)
plot(t,y(:,8)); xlabel('d'); ylabel('cases')
title('Daily Detected Cases'); xlim([56,420]); 
subplot(2,2,4)
imshow('PortugalCorona.png'); axis('off')

%% Prediction
% Simulation future behaviour
tspan = [420,460]; y0 = y8(end,:);
kps=1.8e-2; k=0.3; td=50;
[t9,y9] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% First Simulation (Easter leakage for 25 days)
tspan = [461,485]; y0 = y9(end,:);
kps=27.8; k=0.3; td=50;
[t10,y10] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
tspan = [486,660]; y0 = y10(end,:);
kps=1.8e-2; k=0.3; td=50;
[t11,y11] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Second Simulation (Easter leakage for 50 days)
tspan = [460,510]; y0 = y9(end,:);
kps=27.8; k=0.3; td=50;
[t12,y12] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
tspan = [510,660]; y0 = y12(end,:);
kps=1.8e-2; k=0.3; td=50;
[t13,y13] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Second Simulation (Schools leakage for 25 days)
tspan = [460,485]; y0 = y9(end,:);
kps=1.6; k=0.3; td=50;
[t14,y14] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
tspan = [485,660]; y0 = y14(end,:);
kps=1.8e-2; k=0.3; td=50;
[t15,y15] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
% Second Simulation (Schools leakage for 50 days)
tspan = [460,510]; y0 = y9(end,:);
kps=1.6; k=0.3; td=50;
[t16,y16] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);
tspan = [510,660]; y0 = y16(end,:);
kps=1.8e-2; k=0.3; td=50;
[t17,y17] = ode15s(@(t,y) PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v),tspan,y0,options);

tf1 = [t9;t10;t11];
tf2 = [t9;t12;t13];
tf3 = [t9;t14;t15];
tf4 = [t9;t16;t17];
yf1 = [y9;y10;y11];
yf2 = [y9;y12;y13];
yf3 = [y9;y14;y15];
yf4 = [y9;y16;y17];

figure(2)
subplot(1,2,1)
semilogy(t,y(:,8)); hold on
semilogy(tf1,yf1(:,8)); semilogy(tf2,yf2(:,8)); 
semilogy(tf3,yf3(:,8)); semilogy(tf4,yf4(:,8));
plot([460;460],[1;1e4],'--k');
title('Daily Detected Cases'); xlim([56,660]); ylim([1e0,1e5]);
xlabel('d'); ylabel('cases');
legend('Before Easter','25 Days School and Easter','50 Days School and Easter','25 Days Schools', '50 Days School','Location','southwest')

subplot(1,2,2)
semilogy(t,y(:,5)); hold on;
semilogy(tf1,yf1(:,5)); semilogy(tf2,yf2(:,5)); 
semilogy(tf3,yf3(:,5)); semilogy(tf4,yf4(:,5));
plot([460;460],[1;1e5],'--k');
title('Identified Infected'); xlim([56,660]);ylim([1e0,1e6]);
xlabel('d'); ylabel('cases');
legend('Before Easter','25 Days School and Easter','50 Days School and Easter','25 Days Schools', '50 Days School','Location','southwest')

%% Right Hand Side Function
function dy = PSEIRD(t,y,a,kps,k,ti,tid,d,l,tr,td,krp,v)
    dy(1) = krp*y(6)-kps*y(1)-v+a*(1-y(4)/sum(y(2:7)))*y(5)*y(2);                  % dP/dt
    dy(2) = kps*y(1)-k*y(4)/sum(y(2:7))*y(2)-a*(1-y(4)/sum(y(2:7)))*y(5)*y(2);     % dS/dt
    dy(3) = k*y(4)/sum(y(2:7))*y(2)-1/ti*y(3);                                      % dE/dt
    dy(4) = 1/ti*y(3)-((1-d)/tr+d/tid)*y(4);                                        % dI/dt
    dy(5) = d/tid*y(4)-((1-l)/tr+l/td)*y(5);                                        % dId/dt
    dy(6) = 1/tr*((1-l)*y(5)+(1-d)*y(4))+v-krp*y(6);                                % dR/dt 
    dy(7) = l/td*y(5);                                                              % dD/dt

    % New daily detected cases
    dy(8) = d/(tid*ti)*y(3)-((1-d)/tr+d/tid)*y(8);
    % Total detetcted cases
    dy(9) = d/td*y(5);

    dy = dy';
end