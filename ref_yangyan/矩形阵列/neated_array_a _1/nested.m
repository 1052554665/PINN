function B=nested(n1,n2,d)
% f=1000;
c=340;
% lammda=c/f;
% d=lammda/2;
M_line=n1+n2;
% array_line=[0:d:(n1-1)*d,2*n1*d:(n1+1)*d:n2*(n1+1)*d];
% array_line=[0:d:(n1-1)*d,n1*d:(n1+1)*d:n2*(n1+1)*d];
array_line=[0:d:(n1-1)*d,(2*n1-1)*d:n1*d:(n1*n2+n1-1)*d];

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
