clc,clear,close all

%% Function which we want to make the fourier series of
dx = 0.001;
x = -pi:dx:pi;

f_x = 0*x;
lx = length(x);
i_1 = round(lx/6):round(lx/3);
i_2 = round(lx/3):round(lx*5/6);

f_x(round(lx/6):round(lx/3))= 3*(1:length(i_1));
f_x(round(lx/3):round(lx*5/6))= -3*(1:length(i_2))+f_x(round(lx/3));

%% Plot function
figure('Name','Initial Function');
plot(x,f_x);

%% Compute Fourier Series
% the basis is the following formula
%   #
%   #   f(x) = ∑ a*cos(k*2π*x/L) + b*sin(k*2*π*x/L)
%   #  
% For finding the coeeficients we make the dotproduct of
%   #
%   #   <f(x),cos(k*2π*x/L)> and <f(x),sin(k*2π*x/L)>
%   #
% The first coefficient is gained by the integral of the area devided by
% two

fFc = (sum(f_x.*ones(size(x)))*dx/pi)/2; % First Fourier Coefficient (vertical offset)
FS = fFc; % Fourier Series first only first fourier coefficient for vertical offset
% The Approximation will be done by 50 terms of fourier (not infinite sum)

figure('Name','Function and Fourier Series')
hold on
title('Fourier Series of different orders')
plot(x,f_x,'k','LineWidth',3);

i = 1;
m = [1,5,10,25,50];

for k = 1:50
    a(k) = 1/pi * sum(f_x.*cos(k*x))*dx;
    b(k) = 1/pi * sum(f_x.*sin(k*x))*dx;
    FS = FS + a(k)*cos(k*x) + b(k)*sin(k*x);
    if k == m(i)
    plot(x,FS,'LineWidth',2);
    i=i+1;
    pause(.25);
    end
end
legend('f(x)','order: 1','order: 5','order: 10','order: 25','order: 50')
