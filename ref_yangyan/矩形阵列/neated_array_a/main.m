clear;
clc;
n1=2;
n2=3;
f=[3500 3500];
d=0.072;
fs=16000;
c=340;
B=nested(n1,n2,d);
M=size(B,1);
% theta=[-27.5 -21.5 -10.5 0.5 10 15.5];
% fe=[42.5 38.5 30.5 33 31.5 41];
theta=60;
fe=40;
K=length(theta);
T=1000;
SNR=10;
% Y=data(theta,fe,B,K,T,SNR,f);
x0=2;
y0=2;
z0=0; %阵列中心，即参考点位置
%%麦克风阵列房间设置
r=zeros(M,3);
r(:,1)=x0+B(:,1);
r(:,2)=y0+B(:,2);
r(:,3)=z0+B(:,3);
% r=B;
% s = [x0-l*cosd(fe)*sind(theta) y0-l*cosd(fe)*cosd(theta) z0+l*sind(fe)]            % Source position [x y z] (m)声源位置

% SNR=[-30 -25 -20 -15 -10 -5 0 5 10 15];
% Nk=100;
% for kk=1:length(SNR)
%     sum1=0;
%     for kj=1:Nk
%     soure = [1.8+0.5*rand(1,1) 1.8+0.5*rand(1,1) 3;1.8+0.5*rand(1,1) 1.8+0.5*rand(1,1) 3];

soure = [1.9 1.9 3;2.5 2.5 3];   
C = [5 5 5];                % Room dimensions [x y z] (m)
beta = 0;                 % Reverberation time (s)混响时间
n = 4096;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = 3;                 % -1 equals maximum reflection order!最大反射系数-1
dim = 3;                   % Room dimension房间尺寸
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter
% h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
%%
t=0:1/fs:(n-1)*1/fs;
S(1,:)=1000*cos(2*pi*f(1)*t);
S(2,:)=1000*sin(2*pi*f(2)*t); %声音信号
% S2=10*sin(2*pi*800*t); 
% S=S1+S2;
a=zeros(M,n);
for p=1:length(f)
s=soure(p,:);
h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
for i=1:M
    a(i,:)=fftfilt(h(i,:),S(p,:)); 
    % a(i,:)=awgn(a(i,:), SNR(kk), 'measured');
    Y(i,:)=fft(a(i,:),n);
    i=i+1;
end
Ys=Y(:,round(f(p)*n/fs)+1);  
% 
[J,B_vir,vir_B,vir_Y]=difference(B,Ys,T,f(p),d);
vir_R=vir_Y*vir_Y'/T;
 [U,D,V] = svd(vir_R);  %求R的奇异值分解
 Un = U(:,K+1:end); %噪声子空间
xi=x0+vir_B(:,1);
yi=y0+vir_B(:,2);
zi=z0+zeros(size(vir_B,1),1);
% ang1=0:1:80;  %扫面方位角
% ang2=20:1:60;  %扫描俯仰角
% for k1=1:length(ang1)
%    for k2=1:length(ang2)
%     %a1=exp(-j*2*pi*f*(0:size(Rx1,1)-1)'*d*sind(ang1(k1))*cosd(ang2(k2))/c);
%     %a2=exp(-j*2*pi*f*(0:size(Rx2,1)-1)'*d*cosd(ang1(k1))*cosd(ang2(k2))/c);
%     a=exp(-j*2*pi*f*(xi*sind(ang1(k1))*cosd(ang2(k2))+yi*cosd(ang1(k1))*cosd(ang2(k2))+zi*sind(ang2(k2)))/c);
%     %w1=[a1;a2];
% %     SP(k2,k1)=abs(1/(a'*vir_R*a));  %波束形成
%       SP(k2,k1)=abs(1/(a'*Un*Un'*a));%   music算法
%       A(:,(k1-1)*length(ang2)+k2)=a;
%    end
% end
%%
x2 = x0;
y2 = y0;
z2 = z0;
%-------扫描范围------%
step_x = 0.1;  % 步长设置为0.1
step_y = 0.1;
x = (0:step_x:5);  % 扫描范围 
z = 3;
y = (0:step_y:5);
for k1=1:length(y)
        for k2=1:length(x)
       Ri = sqrt((x(k2)-xi).^2+(y(k1)-yi).^2+(z-zi).^2);
        Ri2 = sqrt((x(k2)-x2).^2+(y(k1)-y2).^2+(z-z2).^2);
        % 该扫描点到各阵元的聚焦距离矢量   一行一行的扫描，把数值再一行一行的赋给Pcbf
        Rn = Ri-Ri2;
        % 扫描点到各阵元与参考阵元的程差矢量
        b = exp(-j*2*pi*f(p)*Rn/c); % 声压聚焦方向矢量
        A(:,(k1-1)*length(x)+k2)=b;
%         SP(k1,k2) = abs(b'*Un*Un'*b); %music
          SP(k1,k2,p) = abs(b'*vir_R*b); % cbf
        end
end

P_sc(:,:,p) = CLEAN_SC(0.9, 100, vir_R,A);
P_sc1(:,:,p)=reshape(abs(P_sc(:,:,p)),[length(x),length(y)]);
  end
   Ps=abs(P_sc1(:,:,1)+P_sc1(:,:,2));
Ps=Ps';
% for k1 = 1:length(y);
%    pp(k1) = max(SP(k1,:)); % Pcbf 的第k1行的最大元素的值
% end
% % for k1 = 1:length(y);
% %    pp1(k1) = max(SP(k1,:,1)); % Pcbf 的第k1行的最大元素的值
% % end
% % for k1 = 1:length(y);
% %    pp2(k1) = max(SP(k1,:,2)); % Pcbf 的第k1行的最大元素的值
% % end
% % SP=SP(:,:,1)/max(pp1)+SP(:,:,2)/max(pp2);
% SP=SP(:,:,1)+SP(:,:,2);
%%  
% SP1=SP(:,:,1)';
% SP2=SP(:,:,2)';
% [X_max1,Y_max1]=find(SP1==max(SP1(:)));
% [X_max2,Y_max2]=find(SP2==max(SP2(:)));
% r_x1(kj)=y(X_max1);
% r_y1(kj)=x(Y_max1);
% r_x2(kj)=y(X_max2);
% r_y2(kj)=x(Y_max2);
% disdence1=(r_x1(kj)-soure(1,1))^2+(r_y1(kj)-soure(1,2))^2+(r_x2(kj)-soure(2,1))^2+(r_y2(kj)-soure(2,2))^2;
% sum1=sum1+disdence1;
%     end
%    RMSE1(kk)=sqrt(sum1/(2*Nk));%SBL算法 
% end


SP=SP(:,:,1)+SP(:,:,2);
for k1 = 1:length(y)
   pp1(k1) = max(SP(k1,:)); % Pcbf 的第k1行的最大元素的值
end
SP=SP/max(pp1);
SP=20*log10(SP);

figure(1)
scatter3(r(:,1),r(:,2),r(:,3),'r*');
hold on;
scatter3(s(:,1),s(:,2),s(:,3),'b*');
% title('声源及麦克风位置');
% xlim([0,7]);
% ylim([0,7]);
% zlim([0,7]);
figure(2)
plot(r(:,1),r(:,2),'r*');
% title('二维嵌套麦克风阵列');
% xlim([0.5,1.5]);
% ylim([0.5,1.5]);
figure(3)
plot(xi,yi,'r*');
% title('虚拟麦克风阵列');
% xlim([0.5,1.5]);
% ylim([0.5,1.5]);
figure(4)
surf(x,y,SP);
xlabel('x(m)');
ylabel('y(m)');
zlim([-30 0]);
% title('三维声源图');
% shading interp;
hold on;
colorbar
clim([-30 0]);
figure(5)
pcolor(x,y,SP);
xlabel('x(m)');
ylabel('y(m)');
zlim([-30 0]);
% title('二维声源图');
shading interp;
hold on;plot(1.8,1.8,'ro','LineWidth',1);
hold on;
plot(2.3,2.3,'ro','LineWidth',1);
hold on;

colorbar
clim([-30 0]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 16);
figure(6)
pcolor(x,y,Ps);
shading interp;
hold on;
% plot(s(1,1),s(1,2),'rp');
xlabel('x(m)');
ylabel('y(m)');
% zlim([-30 0]);
% title('声学成像图')
colorbar

% figure(6)
% plot(SNR,RMSE1,'-rs');hold on;
% h1=plot(SNR,RMSE1,'-rs');hold on;