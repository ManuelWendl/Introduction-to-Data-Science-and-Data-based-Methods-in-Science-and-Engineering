clc, clear, close all

dx = .1;
x = (-2:dx:2)';
y = 3*x + 3*randn(size(x))+5; 

d = [ones(length(x),1), x];

%Pseudo Inverse
[U,S,V] = svd(d,'econ');
m = V*inv(S)*U'*y;

figure('Name','Linear Regression');
hold on
scatter(x,y);
plot(x,d*m,'r');
plot(x,3*x+5,'b--');
xlabel('x')
ylabel('y')
legend('Data','Linear Regression','True Fit');
