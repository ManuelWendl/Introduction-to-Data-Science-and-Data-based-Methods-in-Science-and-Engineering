clc,clear,close all
%% Creating Audio Signal and Noisy set

% time of Audiosequnce
dt = 0.0005;
t = 0:dt:1;

audio = 4*sin(2*pi*100*t) + 8*sin(2*pi*60*t) + 4*cos(2*pi*20*t) +...
    6*cos(2*pi*5*t);
noisy = audio + 10*randn(size(t));

%% Plot Audio signals

figure('Name','Audio Data and Power spectrum');
subplot(2,1,1)
hold on
plot(t,noisy,'r');
plot(t,audio,'k','LineWidth',2);
legend('Noisy Audio','Original Audio');

%% Fast Fourier Transformation 

% FC are the Fourier Coeficients
FC = fft(noisy,length(t));

% Power of different frequencies
P = abs(FC)/length(t);

freq = 1/(dt*length(t))*(1:length(t));

subplot(2,1,2)
hold on
plot(freq,P);
plot(freq,ones(1,length(freq))*1.5,'r')
xlim([1, 300]);
xlabel('frequencies in Hz')
ylabel('Power of frequencies')
title('Powerspectrum')
legend('Powerspectrum','Truncation level')

%% Filter Noisy set

ind = P>1.5;
FCclean = FC.*ind;

audiodenoised = ifft(FCclean);

% Plot denoised and original for comparison
figure('Name','Denoised and Original compared');
hold on
plot(t,audiodenoised,'r','LineWidth',2);
plot(t,audio,'k','LineWidth',2);
legend('Denoised','Original');
title('Overlayed original and denoised audio Signal');



