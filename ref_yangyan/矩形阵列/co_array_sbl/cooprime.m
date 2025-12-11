function B=cooprime(n1,n2,f)
%n1为子阵列1的间距
%n2为子阵列2的间距
Multi=2;%扩展系数
n1_num=n2;
n2_num=Multi*n1;%扩展互质阵列
c=340;
lammda=c/f;
d=lammda/2;
M_line=n1_num+n2_num-1;
array_line=[0:n1:(n1_num-1)*n1,n2:n2:(n2_num-1)*n2];
array_line=sort(array_line);
array_line=array_line*d;
a=[];
b=[];
for k1=1:length(array_line)
    x=array_line';
    a=[a;x];
    y=array_line(k1)*ones(length(array_line),1);
    b=[b;y];
end
 B(:,1)=a;
 B(:,2)=b;
 B(:,3)=zeros(length(a),1);