clc, clear, close all
%% Simulate Data
sigma = 10; rho = 28; beta = 8/3;
options = odeset('MaxStep',1e-3);
y0 = [0 1 1.05]; tspan = [0 80]; 
[t,y] = ode45(@(t,y) F(t, y, sigma, rho, beta), tspan, y0, options);
figure(1)
plot3(y(:,1),y(:,2),y(:,3)); xlabel('x'),ylabel('y'),zlabel('z');

%% Finite Difference Derivatives
yt = differential(y,t);
y = y(2:end-1,:);

%% Set up sparse Linear Solver
PSI = [y,y.^2,y(:,1).*y(:,2),y(:,1).*y(:,3),y(:,2).*y(:,3)];
lambda = 1;
cvx_begin;
    variable eta_x(size(PSI,2)); % sparse vector of coefficients 
    minimize(norm(PSI*eta_x - yt(:,1),2) + lambda * norm(eta_x,1));
cvx_end;
cvx_begin;
    variable eta_y(size(PSI,2)); % sparse vector of coefficients 
    minimize(norm(PSI*eta_y - yt(:,2),2) + lambda * norm(eta_y,1));
cvx_end;
cvx_begin;
    variable eta_z(size(PSI,2)); % sparse vector of coefficients 
    minimize(norm(PSI*eta_z - yt(:,3),2) + lambda * norm(eta_z,1));
cvx_end;
table = table(eta_x,eta_y,eta_z,'VariableNames',{'xi_x','xi_y','xi_z'},'RowNames',...
    {'x','y','z','xx','yy','zz','xy','xz','yz'})

% Simulate discovered system and compare
y0 = [8 5 1]; tspan = [0 20]; 
[t1,y1] = ode45(@(t,y) DiscoveredSystem(t,y,eta_x,eta_y,eta_z), tspan, y0);
[t2,y2] = ode45(@(t,y) F(t, y, sigma, rho, beta), tspan, y0);
y0 = [0 -3 4]; tspan = [0 20]; 
[t3,y3] = ode45(@(t,y) DiscoveredSystem(t,y,eta_x,eta_y,eta_z), tspan, y0);
[t4,y4] = ode45(@(t,y) F(t, y, sigma, rho, beta), tspan, y0);
figure(2)
hold on
plot3(y2(:,1),y2(:,2),y2(:,3),'Color',[0 0.4470 0.7410]); 
plot3(y1(:,1),y1(:,2),y1(:,3),'Color',[0.8500 0.3250 0.0980],'LineStyle','--'); 
plot3(y4(:,1),y4(:,2),y4(:,3),'Color',[0 0.4470 0.7410]); 
plot3(y3(:,1),y3(:,2),y3(:,3),'Color',[0.8500 0.3250 0.0980],'LineStyle','--'); 
plot3(8,5,1,'Color','r','Marker','o')
plot3(0,-3,4,'Color','r','Marker','o')
xlabel('x'),ylabel('y'),zlabel('z'); legend('Discovered Dynamics','Lorenz Dynamics')
%%
figure(3)
label = [{'x(t)'},{'y(t)'},{'z(t)'}];
for i = 1:3
subplot(3,2,i*2-1)
hold on
plot(t2,y2(:,i),'Color',[0 0.4470 0.7410])
plot(t1,y1(:,i),'Color',[0.8500 0.3250 0.0980],'LineStyle','--')
xlabel('t');ylabel(label(i)); legend('Lorenz Dynamics','Discovered Dynamics','Location','southwest')
subplot(3,2,i*2)
hold on
plot(t4,y4(:,i),'Color',[0 0.4470 0.7410])
plot(t3,y3(:,i),'Color',[0.8500 0.3250 0.0980],'LineStyle','--')
xlabel('t');ylabel(label(i)); legend('Lorenz Dynamics','Discovered Dynamics','Location','southwest')
end

%% Discovered system:
function dy = DiscoveredSystem(t,y,eta_x,eta_y,eta_z)
    Psi = [y(1),y(2),y(3),y(1)^2,y(2)^2,y(3)^2,y(1)*y(2),y(1)*y(3),y(2)*y(3)];
    dy = (Psi*[eta_x,eta_y,eta_z])';
end
%% Right hand side Function
function dy = F(t, y, sigma, rho, beta)
% Evaluates the right hand side of the Lorenz system
% x' = sigma*(y-x)
% y' = x*(rho - z) - y
% z' = x*y - beta*z
% typical values: rho = 28; sigma = 10; beta = 8/3;
    dy = zeros(3,1);
    dy(1) = sigma*(y(2) - y(1));
    dy(2) = y(1)*(rho - y(3)) - y(2);
    dy(3) = y(1)*y(2) - beta*y(3);
    return
end
%% Compute Derivative
function dy = differential(y,ydim)
    for i = 2:length(y)-1
        dy(i-1,:) = (y(i+1,:)-y(i-1,:))/(ydim(i+1)-ydim(i-1));
    end
end
