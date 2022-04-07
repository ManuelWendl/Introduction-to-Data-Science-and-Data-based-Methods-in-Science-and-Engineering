clc, clear, close all

dx = .5;
x = (-6:dx:7)';

y = x.^3 + 50*randn(size(x)) + 50;

x1 = [ones(size(x,1),1), x];
x2 = [ones(size(x,1),1), x, x.^2];
x3 = [ones(size(x,1),1), x, x.^2, x.^3];
x10 = [ones(size(x,1),1), x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8,...
        x.^9,x.^10];

data = {x1,x2,x3,x10};
tag = {'Linear Regression', 'Quadratic Regression', 'Cubic Regression', ...
        'Decic Regression'};

figure('Name','Polynomial Regression')
for i=1:4
subplot(2,2,i)
hold on 
title(tag{i})
scatter(x,y,'.')
[U,S,V] = svd(data{i},'econ');
c = V*inv(S)*U'*y;
plot(x,data{i}*c,'r')
xlim([-6, 7])
legend('Data','Regression')
end