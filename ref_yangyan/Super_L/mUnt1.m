clear all
close all
clc
rad=pi/180;
T=1000; %快拍数
n1=3;
n2=3;
c=340;%声速
f=5000;% 感兴趣的频率
lammda = c/f;
d=lammda/2;
theta=[30 40 65 100 130] ;  %方位角
fe=[25 45 60 63 50] ;  %俯仰角
% theta=[37 117];
% fe=[25 63];
K=length(theta);
w=2*pi*f;
SNR=20;
[B,array_num,array_int]=Spuer_nested_L(n1,n2);
Y=data(theta,fe,B*d,K,T,SNR,f);
[J_x,B_vir_x,B_vir1_x,vir_Y_x,vir_Y1_x]=difference(array_int*d,Y(1:array_num,:),T,f,d);
[J_y,B_vir_y,B_vir1_y,vir_Y_y,vir_Y1_y]=difference(array_int*d,Y(array_num+1:2*array_num,:),T,f,d);
vir_Y=[vir_Y_x;vir_Y_y];
R=vir_Y*vir_Y';

[U,D,V] = svd(R);  %求R的奇异值分解
Un = U(:,K+1:end); %噪声子空间
%虚拟阵列坐标
xi=[B_vir1_x*d,zeros(1,length(B_vir1_x))]';
yi=[zeros(1,length(B_vir1_y)),B_vir1_x*d]';
zi=zeros(length(xi),1);

B_t(:,1)=xi;
B_t(:,2)=yi;
B_t(:,3)=zi;
% Y_t=data(theta,fe,B_t,K,T,SNR,f);
% R_t=Y_t*Y_t'/T;
ang1=0:1:180;  %扫面方位角
ang2=0:1:90;  %扫描俯仰角
for k1=1:length(ang1)
   for k2=1:length(ang2)
    %a1=exp(-j*2*pi*f*(0:size(Rx1,1)-1)'*d*sind(ang1(k1))*cosd(ang2(k2))/c);
    %a2=exp(-j*2*pi*f*(0:size(Rx2,1)-1)'*d*cosd(ang1(k1))*cosd(ang2(k2))/c);
    a=exp(-j*2*pi*f*(xi*sind(ang1(k1))*cosd(ang2(k2))+yi*cosd(ang1(k1))*cosd(ang2(k2))+zi*sind(ang2(k2)))/c);
    %w1=[a1;a2];
    A(:,k1*k2+k2)=a;
    Pcbf(k2,k1)=abs(a'*R*a);
    SP(k2,k1)=abs(1/(a'*Un*Un'*a));
    % P_t(k2,k1)=abs(a'*R_t*a);
   end
end
SP=SP/max(max(SP));
SP=20*log10(SP);

figure(1);
plot(B(:,1),B(:,2),'r*');
% hold on;
% scatter3(s(:,1),s(:,2),s(:,3),'b*');
% xlim([x0-4 x0+4]);
% ylim([y0-4 y0+4]);
% zlim([0,2*z0]);
figure(2);
xj=[B_vir_x*d,zeros(1,length(B_vir_x))];
yj=[zeros(1,length(B_vir_y)),B_vir_y*d];
plot(xj,yj,'r*');
figure(3);
plot(xi,yi,'r*');
figure(4);
surf(ang1,ang2,Pcbf);
xlabel('方位角/degree'),ylabel('俯仰角/degree')
title('三维声源图')
colorbar
xlabel('elevation(degree)')
ylabel('azimuth(degree)')
zlabel('magnitude(dB)')
% [X_max,Z_max]=find(SP==max(SP(:)));
figure(5)
pcolor(ang1,ang2,SP);
shading interp;
% text(ang1(Z_max)+2,ang2(X_max)+2, ['(',num2str(ang1(Z_max)),',',num2str(ang2(X_max)),')'],'color','r');
xlabel('方位角/degree');
ylabel('俯仰角/degree');
title('虚拟阵列声源图')
colorbar
% clim([-30 0]);
% figure(6)
% pcolor(ang1,ang2,P_t);
% shading interp;
% xlabel('方位角/degree');
% ylabel('俯仰角/degree');
% title('理论阵列声源图')
% colorbar