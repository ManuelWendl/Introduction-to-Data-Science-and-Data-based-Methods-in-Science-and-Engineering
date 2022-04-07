clc, clear, close all
%% Import Data
import = open('HUDRAT2_BestTrack.mat');
import = import.Hudrat2BestTrack;

%% Preprocess Data
import(isnan(import))=0;

j = 1;

for i=1:size(import,1)
    if import(i,3)>11
        z = i+8;
       while import(z,3)<1
           if z==size(import,1)
               break;
           end
           data(j,:) = [import(i+3),...
               import(z,5),import(z,6),import(z,7),...
               import(z-1,5),import(z-1,6),import(z-1,7),...
               import(z-2,5),import(z-2,6),import(z-2,7),...
               import(z-3,5),import(z-3,6),import(z-3,7),...
               import(z-4,5),import(z-4,6),import(z-4,7),...
               import(z-5,5),import(z-5,6),import(z-5,7),...
               import(z-6,5),import(z-6,6),import(z-6,7),...
               import(z-7,5),import(z-7,6),import(z-7,7)];
           j=j+1;
           z=z+1;
       end
    end
end


for i = 1:size(data,1)
    Date = mod(data(i,1),10000);
    Day = mod(data(i,1),100);
    Month = (Date-Day)/100;
    data(i,1) = 30*(Month-1)+Day;
end