clear all
close all
clc
f=1000;% 感兴趣的频率
fs=16000;%采样频率
c=340;%声速
w=2*pi*f;
l=20; %声源到阵列的距离
dis_sub_array_1 = 2;%%  M阵元数量多，阵元间距短的子阵的间距
dis_sub_array_2 = 3;%%  N阵元数量少，阵元间距大的子阵的间距
multi = 2;%%阵列数量少的子阵的扩展倍数
sub_array_1_num = dis_sub_array_2;% N
sub_array_2_num = dis_sub_array_1;% M
sub_array_2_num = sub_array_2_num*multi;% multi*M
array_num = sub_array_1_num+sub_array_2_num-1;% multi*M+N-1 阵元数
M=2*array_num;%总阵元数
lambda = c/f;%%波长
d = 0.5*lambda;%最小阵元间隔
array_structure = [0,dis_sub_array_1:dis_sub_array_1:(sub_array_1_num-1)*dis_sub_array_1,dis_sub_array_2:dis_sub_array_2:(sub_array_2_num-1)*dis_sub_array_2];
array_structure=sort(array_structure,"ascend");
array_structure1 = d*array_structure;%%阵列真实结构构造   对数组中的数升序排序
W = zeros(array_num,array_num);  %定义一个阵元数x阵元数的全零矩阵
%array_structure = array_structure/d;  %阵列结构归一化
%求差分共阵
for ik = 1:array_num
    for jk = 1:array_num
        W(ik,jk) = array_structure(ik)-array_structure(jk);
    end
end
W = reshape(W,1,[]);%%虚拟阵列阵元分布
max_flag = round(max(W));
min_flag = round(min(W));
histog = zeros(1,max_flag-min_flag+1);
for ik = 1:length(W)
    histog(round(W(ik))+abs(min_flag)+1) = histog(round(W(ik))+abs(min_flag)+1)+1;
end%%
%%虚拟权值函数构建
J = [];
for ik = min(W):max(W)
    II = zeros(array_num,array_num);
   for jk = 1:array_num
        for kk = 1:array_num
           if array_structure(jk) - array_structure(kk) == ik    %%如果Pi-Pj=Pk,则II(i,j)=1/histog(k),矩阵II的其他项为0
                II(jk,kk) = 1/(histog(ik+abs(min(W))+1));
            end
        end
    end
    J = [J;reshape(II,1,[])];%%冗余阵元整合矩阵构建                        %向量化II，将其作为J的第k行。
end
B=min(W):max(W);
histog_flag = round(abs(min_flag)+1);  %histog是对称的，找出中心位置
%histog_f=round(length(B)/2);
for ik = 0:min(abs(min(W)),abs(max(W)))
    if histog(histog_flag+ik) == 0 || histog(histog_flag-ik) == 0
        flag = ik-1;
        break;
    end
end  %找到连续虚拟阵元位置flag
B1=B(histog_flag-flag:histog_flag+flag);
% B2=[B1(1:flag) B1(flag+2:2*flag+1)];
xi=[B1*d,zeros(1,length(B1))]';
yi=[zeros(1,length(B1)),B1*d]';
zi=zeros(length(xi),1);
ikk=1:30;
theta=0+60*rand(1,length(ikk));
fe=30+30*rand(1,length(ikk));
for kkk=1:length(ikk)
%%麦克风阵列房间设置
r=zeros(M,3);
r(:,1)=[array_structure1,zeros(1,(array_num))];
r(:,2)=[zeros(1,(array_num)),array_structure1];
r(:,3)=zeros(M,1);
%s=[0 1 6];
%s(:,1)=l.*cosd(theta).*sind(fe);
%s(:,2)=l.*sind(theta).*sind(fe);
%s(:,3)=l.*cosd(fe);
s = [-l*cosd(fe(kkk))*sind(theta(kkk)) -l*cosd(fe(kkk))*cosd(theta(kkk)) l*sind(fe(kkk))];             % Source position [x y z] (m)声源位置
C = [30 30 30];                % Room dimensions [x y z] (m)
beta = 0;                 % Reverberation time (s)混响时间
n = 4096;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = 3;                 % -1 equals maximum reflection order!最大反射系数-1
dim = 3;                    % Room dimension房间尺寸
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter
h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
t=0:1/fs:(n-1)*1/fs;
S=10*cos(2*pi*f*t)+sin(2*pi*200*t); %声音信号
a=zeros(M,n);
for i=1:M
    a(i,:)=fftfilt(h(i,:),S);
    Y(i,:)=fft(a(i,:),n);
    i=i+1;
end
L=n;
Ys=Y(:,round(f*n/fs)+1);  % 选择600Hz频率点：(n-1)*fs/N=600Hz,n为600Hz对应的采样点,即n=f*N/fs+1
%Rxx=Ys*Ys'/L;   %自相关函数（协方差矩阵
R1=Ys(1:array_num)*Ys(1:array_num)'/L;%阵列1的协方差矩阵
z1=reshape(R1,[],1); %阵列1协方差矩阵向量化
v1 = J*z1;%%去冗余之后的信号的表达式
%B=sort(unique(W));
vir_array1 = v1(histog_flag-flag:histog_flag+flag);%%连续虚拟阵列孔径划分
Rx1=vir_array1*vir_array1';
%Rx1= toeplitz(v1(histog_flag:histog_flag+flag,1),flipud(v1(histog_flag-flag:histog_flag,1)));%%向量托普利茨化
%[Ux,Dx,Vx] = svd(Rx1);  %求R的奇异值分解
%Unx = Ux(:,source_num+1:end); %噪声子空间
R2=Ys(array_num+1:M)*Ys(array_num+1:M)'/L;%阵列2的协方差矩阵
z2=reshape(R2,[],1);%阵列2协方差矩阵向量化
%%
v2=J*z2;
vir_array2=v2(histog_flag-flag:histog_flag+flag);
% vir_array2=[vir_array2(1:flag);vir_array2(flag+2:2*flag+1)];
Rx2=vir_array2*vir_array2';
%Rx2= toeplitz(v2(histog_flag:histog_flag+flag,1),flipud(v2(histog_flag-flag:histog_flag,1)));%%向量托普利茨化
vir_array=[vir_array1;vir_array2];  %Y
R=vir_array*vir_array';
ys=[real(vir_array);imag(vir_array)];
%deg = -90:90; %角度划分
%kkk=0;
ang1=0:1:60;  %扫面方位角
ang2=30:1:60;  %扫描俯仰角
%a1=zeros(length(vir_array1),length(ang1)*length(ang2));
%a2=zeros(length(vir_array2),length(ang1)*length(ang2));
for k1=1:length(ang2)
    for k2=1:length(ang1)
% a1(:,(k1-1)*length(ang1)+1:k1*length(ang1))=exp(-j*2*pi*f*(0:size(Rx1,1)-1)'*d*sind(ang1)*cosd(ang2(k1))/c);
% a2(:,(k1-1)*length(ang1)+1:k1*length(ang1))=exp(-j*2*pi*f*(0:size(Rx2,1)-1)'*d*cosd(ang1)*cosd(ang2(k1))/c);
% A=[a1;a2];
a=exp(-j*2*pi*f*(xi*sind(ang1(k2))*cosd(ang2(k1))+yi*cosd(ang1(k2))*cosd(ang2(k1))+zi*sind(ang2(k1)))/c);
A(:,(k1-1)*length(ang1)+k2)=a;
    end
end
As=[real(A.'),imag(A.')].';
lambda0 = 1e-2; 
learn_Lambda0 = 1; 
[Weight0,gamma_est0,gamma_used0,count0] = MSBL(As,ys, lambda0, learn_Lambda0);
Gamma=diag(gamma_est0);
N=length(Weight0);
Weight=reshape(Weight0,[length(ang1),length(ang2)]);
SP=Weight';
[X_max,Z_max]=find(SP==max(max(SP)));
RMSE(kkk)=sqrt(((theta(kkk)-ang1(Z_max))^2+(fe(kkk)-ang2(X_max))^2)/2);
r_theta(kkk)=ang1(Z_max);
r_fe(kkk)=ang2(X_max);
end
% figure(1);
% surf(ang1,ang2,abs(SP));
% xlabel('方位角/degree'),ylabel('俯仰角/degree'),zlabel('magnitude(dB)')
% title('三维单声源图')
% colorbar
% figure(2);
% pcolor(ang1,ang2,abs(SP));
% hold on;
% shading interp;
% text(ang1(Z_max)+2,ang2(X_max)+2, ['(',num2str(ang1(Z_max)),',',num2str(ang2(X_max)),')'],'color','r');
% xlabel('方位角/degree');
% ylabel('俯仰角/degree');
% title('声源图')
% colorbar
figure(3)
plot(ikk,theta,'r*',ikk,r_theta,'go');
ylabel('方位角');
figure(4)
plot(ikk,fe,'r*',ikk,r_fe,'go');
ylabel('俯仰角');
figure(5)
plot(ikk,RMSE,'b*');
ylabel('RMSE'); 