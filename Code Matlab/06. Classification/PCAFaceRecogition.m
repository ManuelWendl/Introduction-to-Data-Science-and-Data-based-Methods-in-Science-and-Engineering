clc, clear, close all
%% Import Data
m = 192;
n = 168;
for i=1:37
   for j=1:60 
        if i<10
            url = ['YaleFaces/yaleB0',num2str(i),'/','Picture_',num2str(j),'.pgm'];
            face = importdata(url);
            allfaces(:,(i-1)*60+j) = reshape(face,m*n,1);
        else
            url = ['YaleFaces/yaleB',num2str(i),'/','Picture_',num2str(j),'.pgm'];
            face = importdata(url);
            allfaces(:,(i-1)*60+j) = reshape(face,m*n,1);
        end
   end
end

%% Visualization Prints
figure('Name','Initial Face Plots')
subplot(1,2,1)
hold on
firstimage = ones(6*m,6*n);
count = 0;
for i = 1:6
    for j = 1:6
        firstimage((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(allfaces(:,count*60+1),m,n);
        count = count+1;
    end
end
imshow(mat2gray(firstimage));

subplot(1,2,2)
hold on
lightings = ones(7*m,8*n);
count = 1;
for i = 1:7
    for j = 1:8
        lightings((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(allfaces(:,count),m,n);
        count = count+1;
    end
end
imshow(mat2gray(lightings));

%% Keep Images for Recognition
% Two pictures of every person
count = 1;
for i = 0:36
    ind = randi(60,1,2);
    testfaces(:,[count, count+1]) = double(allfaces(:,i*58+ind));
    allfaces(:,i*58+ind) = [];
    count = count+2;
end

X = double(allfaces);

%% Average (Mean) Face
M = mean(X,2);
figure('Name','Mean Face')
hold on
imshow(uint8(reshape(M,m,n)));
title('Average Face')

B = X - M*ones(1,size(X,2));

%% SVD and Eigenbasis
[U,S,V] = svd(B,'econ');

% Principal Components in colummns of U (eigenfaces)
PC = U;
% Principal Components in V (B*V for same dimensions)
%PC = B*V;

PCs = ones(5*m,5*n);
count = 1;
for i = 1:5
    for j = 1:5
        PCs((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(PC(:,count),m,n);
        count = count+1;
    end
end
figure('Name','Principal Components')
hold on
imshow(mat2gray(PCs));
title('First 25 principal components')

%% Projection onto principal components

person1 = 2;
person2 = 8;

images1 = double(allfaces(:,(person1-1)*58+1:(person1-1)*58+58));
images2 = double(allfaces(:,(person2-1)*58+1:(person2-1)*58+58));

figure('Name','Projection onto Principal components')
subplot(1,3,1)
images1m = ones(8*m,7*n);
count = 1;
for i = 1:8
    for j = 1:7
        images1m((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(images1(:,count),m,n);
        count = count+1;
    end
end
imshow(mat2gray(images1m));
title(['Pictures of person ', num2str(person1)])

subplot(1,3,3)
images2m = ones(8*m,7*n);
count = 1;
for i = 1:8
    for j = 1:7
        images2m((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(images2(:,count),m,n);
        count = count+1;
    end
end
imshow(mat2gray(images2m));
title(['Pictures of person ', num2str(person2)])

PCchosen = [5, 6, 7];

Proj1 = (PC(:,PCchosen))'*images1;
Proj2 = (PC(:,PCchosen))'*images2;

subplot(1,3,2)
hold on
plot3(Proj1(1,:),Proj1(2,:),Proj1(3,:),'rd','LineWidth',4);
plot3(Proj2(1,:),Proj2(2,:),Proj2(3,:),'bd','LineWidth',4);
view([140,20]);
xlabel(['Principal Component ',num2str(PCchosen(1))])
ylabel(['Principal Component ',num2str(PCchosen(2))])
zlabel(['Principal Component ',num2str(PCchosen(3))])
grid('on')
legend(['Person ', num2str(person1)],['Person ', num2str(person2)])

%% Person Mean Face
for i=0:36
    meanFaces(:,i+1)= mean(X(:,i*58+1:i*58+58),2);
end
figure('Name','Mean face of each person')
meanFacesm = ones(6*m,6*n);
count = 1;
for i = 1:6
    for j = 1:6
        meanFacesm((i-1)*m+1:i*m,(j-1)*n+1:j*n) = reshape(meanFaces(:,count),m,n);
        count = count+1;
    end
end
imshow(mat2gray(meanFacesm));
title('Mean face of each person')

%% Choose Principal Components for Prediction

figure('Name','Choosing Principal Components')
subplot(2,2,1)
semilogy(diag(S),'-o');
xlim([0,400]);
title('Singular Values \sigma')

subplot(2,2,2)
lambda = diag(S).^2;
for i = 1:length(lambda)
    var(i) = sum(lambda(1:i))/sum(lambda);
end
plot(var,'-o')
xlim([0,400]);
title('Reached variance with k \sigma')

PCchosen1 = [1, 2, 3];
Proj1 = (PC(:,PCchosen1))'*meanFaces;
subplot(2,2,3)
plot3(Proj1(1,:),Proj1(2,:),Proj1(3,:),'rd','LineWidth',4);
xlabel(['Principal Component ',num2str(PCchosen1(1))])
ylabel(['Principal Component ',num2str(PCchosen1(2))])
zlabel(['Principal Component ',num2str(PCchosen1(3))])

PCchosen2 = 5:25;
Proj = (PC(:,PCchosen2))'*meanFaces;
subplot(2,2,4)
plot3(Proj(1,:),Proj(2,:),Proj(3,:),'rd','LineWidth',4);
xlabel(['Principal Component ',num2str(PCchosen2(1))])
ylabel(['Principal Component ',num2str(PCchosen2(2))])
zlabel(['Principal Component ',num2str(PCchosen2(3))])

%% Face Recognition 

Testpicture = 20;

ProjTest = (PC(:,PCchosen2))'*testfaces(:,Testpicture);

ProjDiff = Proj - ProjTest*ones(1,size(Proj,2));

ProjDiff = sum(abs(ProjDiff),1);
[minimum, ind] = min(ProjDiff);

figure('Name','Prediction')
subplot(1,3,1)
imshow(mat2gray(reshape(testfaces(:,Testpicture),m,n)));
title(['Testimage of Person: ',num2str(round(Testpicture/2))])
subplot(1,3,2)
imshow(mat2gray(reshape(meanFaces(:,ind),m,n)));
title(['Recognized Person: ',num2str(ind)])
subplot(1,3,3)
hold on
images = double(allfaces(:,(round(Testpicture/2)-1)*58+1:(round(Testpicture/2)-1)*58+58));
Projp = (PC(:,PCchosen))'*images;
plot3(Projp(1,:),Projp(2,:),Projp(3,:),'kd','LineWidth',4);
plot3(ProjTest(1,:),ProjTest(2,:),ProjTest(3,:),'rd','LineWidth',4);
xlabel(['Principal Component ',num2str(PCchosen1(1))])
ylabel(['Principal Component ',num2str(PCchosen1(2))])
zlabel(['Principal Component ',num2str(PCchosen1(3))])
view([120,30])

count = 0;
for i = 1:size(testfaces,2)
    ProjTest = (PC(:,PCchosen2))'*testfaces(:,i);
    ProjDiff = Proj - ProjTest*ones(1,size(Proj,2));
    ProjDiff = sum(abs(ProjDiff),1);
    [minimum, ind] = min(ProjDiff);
    if ind == round(i/2)
        count = count+1;
    end
end
sprintf('Performance of the recognition algorithm: %f',count/size(testfaces,2))
