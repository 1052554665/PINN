function [B,array_num,array_int]=Spuer_nested_L(n1,n2)
%输出
%B为嵌套阵列坐标
%c为嵌套阵列所在圆的坐标
%输入：n1，n2为两个子阵列数，d为阵元间距，R为圆半径
M_line=n1+2*n2+1;
% d=2*pi*R/((M_line^2-1)/2);
% d=2*pi*R/(n1*n2+n1-1);
% array_line=[0:d:(n1-1)*d,(2*n1-1)*d:n1*d:(n1*n2+n1-1)*d];
% array_line=[0:d:n1*d,(n1+1)*d:(n1+1)*d:n2*(n1+1)*d];
array_int =[0:2:n1-1,n1:n1:n2*n1,n2*n1+2*(n1+1),n2*n1+4*(n1+1):n1+2:n2*n1+4*(n1+1)+(n2-1)*(n1+2),n2*n1+4*(n1+1)+(n2-1)*(n1+2)+2];
B(:,1)=[array_int,zeros(1,length(array_int))];
B(:,2)=[zeros(1,length(array_int)),array_int];
B(:,3)=zeros(2*length(array_int),1);
array_num=length(array_int);