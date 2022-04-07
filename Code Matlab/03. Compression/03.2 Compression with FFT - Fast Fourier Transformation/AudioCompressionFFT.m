clc, clear, close all
%% Import Data
[audio, FS] = audioread('Für-Elise.mp3');
audio = audio(1:300000,1);

%% FFT Data
FC = fft(audio);

%% IFFT Data
perc = [.99 .3 .1 .01 .001];

[FCsort, ind] = sort(FC,'descend');

figure('Name','Compressed Data')
c = [0 0.4470 0.7410];
for i = 1:length(perc)
    limit = abs(FCsort(round(perc(i)*length(audio))));
    clr = abs(FC)>limit;
    FCred = FC.*clr;
    
    audiocomp = ifft(FCred);
    
    %sound(audiocomp,FS)
    
    subplot(2,5,i)
    plot(1:length(audio),real(audiocomp))
    axis([1 length(audio) -1 1])
    title(['compressed to ', num2str(perc(i)*100),' %'])
    
    subplot(2,5,i+5)
    bar(abs(fftshift(FCred)),'FaceColor',c,'EdgeColor',c)
    set(gca,'YScale','log')
    xlim([1 length(audio)])
    ylim([1e-3 1e+4])
    %pause(length(audio)/FS + 0.5)
end

%% Correlation Notes and Frequencies

figure('Name','Correlation Notes and Frequencies')
subplot(2,1,1)
imshow(imread('NotesFür-Elise.jpg'))
title('Tones')

dF = FS/length(audio);                      % hertz
f = -FS/2:dF:FS/2-dF;

subplot(2,1,2)
hold on
axis([0 3000 0 2200])
plot(329.427,2099.4,'or','LineWidth',3)
plot(439.677,1119.72,'ob','LineWidth',3)
plot(494.214,1116.68,'og','LineWidth',3)
plot(621.81,1091.37,'oy','LineWidth',3)
plot(f,abs(fftshift(FC)),'LineWidth',2,'Color',c)
plot(ones(1,2200)*329.6,1:2200,'k')
plot(ones(1,2200)*392,1:2200,'k')
plot(ones(1,2200)*493.8,1:2200,'k')
plot(ones(1,2200)*587.3,1:2200,'k')
plot(ones(1,2200)*698.5,1:2200,'k')
xlabel('frequencies in Hz')
ylabel('|f| Power of frequency')
title('Power Spectrum')
legend('E4','A4','B4','Dis5','Power Spectrum')
