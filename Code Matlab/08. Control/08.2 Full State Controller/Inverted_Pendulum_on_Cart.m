clc, clear, close all
%% Initialise Parameters
m = 1; M = 5; L = 3; g = -9.81; d = 1;
b = 1; % Pendulum up (b=1)

%% Linearised Dynamics
A = [0 1 0 0;
    0 -d/M b*m*g/M 0;
    0 0 0 1;
    0 -b*d/(M*L) -b*(m+M)*g/(M*L) 0];
B = [0; 1/M; 0; b/(M*L)];

% Check for being an unstable critical point:
lamda = eig(A);

%% Design LQR controller
Q = eye(4); R = .0001; Kopt = lqr(A,B,Q,R); Krand = [-50,-60,1500,700];
%% Simulate closed Loop Controller
tspan = 0:.1:14;
x0 = [2;0;pi+.4; 0];   % initial condition
wr = [1;0;pi;0];        % refernece position
uopt = @(x) -Kopt*(x-wr);   % optimal closed loop controller
urand = @(x) -Krand*(x-wr);   % closed loop controller

[topt,xopt] = ode15s(@(t,x) pendulum(x,m,M,L,g,d,uopt(x)),tspan,x0);
[trand,xrand] = ode15s(@(t,x) pendulum(x,m,M,L,g,d,urand(x)),tspan,x0);

figure(1);
plot(topt,xopt); legend('x','v','\phi','\theta')
hold on
plot(trand,xrand); legend('x','v','\phi','\theta')

%% Animation
video = VideoWriter('animation','MPEG-4');
video.FrameRate = 100;
open(video);

figure(2)
for i = 1:length(topt)
    subplot(2,1,1)
    X = xopt(i,1); theta = xopt(i,3);
    plot([X X],[0 4],'LineStyle','--','Color',[.75 .75 .75])
    hold on
    plot([X X+L*sin(pi-theta)],[0 L*cos(pi-theta)],'LineWidth',3,'Color','k');
    rectangle('Position',[X-2,-2,4,2],'FaceColor',[.7,.7,.7]);
    viscircles([X+L*sin(pi-theta),L*cos(pi-theta)],.2,'Color','r');
    xlim([-5,5]); ylim([-2,5]); title('Optimal controller')
    hold off

    subplot(2,1,2)
    X = xrand(i,1); theta = xrand(i,3);
    plot([X X],[0 4],'LineStyle','--','Color',[.75 .75 .75])
    hold on
    plot([X X+L*sin(pi-theta)],[0 L*cos(pi-theta)],'LineWidth',3,'Color','k');
    rectangle('Position',[X-2,-2,4,2],'FaceColor',[.7,.7,.7]);
    viscircles([X+L*sin(pi-theta),L*cos(pi-theta)],.2,'Color','r');
    xlim([-5,5]); ylim([-2,5]); title('Arbitrary stable controller')
    hold off
    f = getframe(gcf);
    writeVideo(video,f);
    pause(.01);
end
video.close();
sprintf('Video Captured')


%% Pendulum Dynamics
function dx = pendulum(x,m,M,L,g,d,u)
    D = m*L^2*(M+m*(1-cos(x(3))^2)); % Denominator

    dx(1,1) = x(2);
    dx(2,1) = 1/D*(-m^2*L^2*g*cos(x(3))*sin(x(3))+m*L^2*(m*L*x(4)^2*sin(x(3))-d*x(2)))+1/D*m*L^2*u;
    dx(3,1) = x(4);
    dx(4,1) = 1/D*((m+M)*m*g*L*sin(x(3))-m*L*cos(x(3))*(m*L*x(4)^2*sin(x(3))-d*x(2)))-1/D*m*L*cos(x(3))*u;
end