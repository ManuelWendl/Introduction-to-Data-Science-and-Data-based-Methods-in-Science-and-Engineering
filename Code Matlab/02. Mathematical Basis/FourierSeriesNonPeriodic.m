clc, clear, close all;

x = -pi:0.1:pi;
y = ones(size(x));
y(1:30) = -1;
y(31:60) = 1;

fft1 = fft(y);
rec1 = ifft(fft1);

figure(1)
hold on
plot(x,y)
plot(x,rec1)