clear,clc,close all
%% Create sphere
e1 = [3;0;0];
e2 = [0;3;0];
e3 = [0;0;3];

[x,y,z] = sphere(50);

figure
subplot(2,4,1)
hold on
quiver3(0,0,0,e1(1),e1(2),e1(3),0,'r');
quiver3(0,0,0,e2(1),e2(2),e2(3),0,'b');
quiver3(0,0,0,e3(1),e3(2),e3(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surface(x,y,z);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('Unit Sphere')

%% Rotation and stretching matrices;
a1 = pi/15;
a2 = -pi/9;
a3 = -pi/20;

Rx = [1 0 0
    0 cos(a1) -sin(a1)
    0 sin(a1) cos(a1)];

Ry = [ cos(a2) 0 -sin(a2)
    0 1 0
    sin(a2) 0 cos(a3)];

Rz = [cos(a3) -sin(a3) 0
    sin(a3) cos(a3) 0
    0 0 1];

E = diag([2;1;0.5]);

R =  Rz*Ry*Rx;
A = R*E

%% tarnsform sphere
xf = 0*x;
yf = 0*y;
zf = 0*z;

for i=1:51
    for ii=1:51
        F = A*[x(i,ii);y(i,ii);z(i,ii)];
        xf(i,ii) = F(1);
        yf(i,ii) = F(2);
        zf(i,ii) = F(3);
    end
end

e1f = A*e1;
e2f = A*e2;
e3f = A*e3;

subplot(2,4,4)
hold on
quiver3(0,0,0,e1f(1),e1f(2),e1f(3),0,'r');
quiver3(0,0,0,e2f(1),e2f(2),e2f(3),0,'b');
quiver3(0,0,0,e3f(1),e3f(2),e3f(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surf(xf,yf,zf);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('A \cdot Unit Sphere')

%% SVD of A

[U,S,V] = svd(A);

subplot(2,4,5)
hold on
quiver3(0,0,0,e1(1),e1(2),e1(3),0,'r');
quiver3(0,0,0,e2(1),e2(2),e2(3),0,'b');
quiver3(0,0,0,e3(1),e3(2),e3(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surface(x,y,z);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('Unit Sphere')

for i=1:51
    for ii=1:51
        F = U*[x(i,ii);y(i,ii);z(i,ii)];
        xf(i,ii) = F(1);
        yf(i,ii) = F(2);
        zf(i,ii) = F(3);
    end
end

e1f = U*e1;
e2f = U*e2;
e3f = U*e3;

subplot(2,4,6)
hold on
quiver3(0,0,0,e1f(1),e1f(2),e1f(3),0,'r');
quiver3(0,0,0,e2f(1),e2f(2),e2f(3),0,'b');
quiver3(0,0,0,e3f(1),e3f(2),e3f(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surf(xf,yf,zf);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('U \cdot Unit Sphere')

for i=1:51
    for ii=1:51
        F = U*S*[x(i,ii);y(i,ii);z(i,ii)];
        xf(i,ii) = F(1);
        yf(i,ii) = F(2);
        zf(i,ii) = F(3);
    end
end

subplot(2,4,7)
hold on
quiver3(0,0,0,e1f(1),e1f(2),e1f(3),0,'r');
quiver3(0,0,0,e2f(1),e2f(2),e2f(3),0,'b');
quiver3(0,0,0,e3f(1),e3f(2),e3f(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surf(xf,yf,zf);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('U \cdot S\cdot Unit Sphere')

for i=1:51
    for ii=1:51
        F = U*S*V'*[x(i,ii);y(i,ii);z(i,ii)];
        xf(i,ii) = F(1);
        yf(i,ii) = F(2);
        zf(i,ii) = F(3);
    end
end

e1f = U*V'*e1;
e2f = U*V'*e2;
e3f = U*V'*e3;

subplot(2,4,8)
hold on
quiver3(0,0,0,e1f(1),e1f(2),e1f(3),0,'r');
quiver3(0,0,0,e2f(1),e2f(2),e2f(3),0,'b');
quiver3(0,0,0,e3f(1),e3f(2),e3f(3),0,'g');
axis equal
axis([-3,3,-3,3,-3,3]);
s = surf(xf,yf,zf);
xlabel('X')
ylabel('Y')
zlabel('Z')
view([45 26]);
title('U \cdot S \cdot V^T \cdot Unit Sphere')
