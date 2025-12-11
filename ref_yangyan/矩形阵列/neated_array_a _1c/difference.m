function [J,B_vir,vir_B,vir_Y]=difference(array,Y,T,f,d)
c=340;
lammda=c/f;
% d=lammda/2;
% d=0.02;
array_num=size(array,1);
array=round(array/d);
array=array(:,1:2);
% W = zeros(array_num,array_num);  %定义一个阵元数x阵元数的全零矩阵
%array_structure = array_structure/d;  %阵列结构归一化
%求差分共阵
% for ik = 1:array_num
%     for jk = 1:array_num
%         W(ik,jk) = array(ik)-array(jk);
%     end
% end
% W = reshape(W,1,[]);%%虚拟阵列阵元分布
for ik=1:array_num
    for jk=1:array_num
        W((ik-1)*array_num+jk,:)= array(ik,:)-array(jk,:);
    end
end
max_flag = round(max(W(:,1)));
min_flag = round(min(W(:,1)));
% histog = zeros(1,max_flag-min_flag+1);
histog=zeros(max_flag-min_flag+1,max_flag-min_flag+1);
% for ik = 1:length(W)
%     histog(round(W(ik))+abs(min_flag)+1) = histog(round(W(ik))+abs(min_flag)+1)+1;
% end%%
for ik = 1:size(W,1)
   histog(W(ik,1)+abs(min_flag)+1,W(ik,2)+abs(min_flag)+1) = histog(W(ik,1)+abs(min_flag)+1,W(ik,2)+abs(min_flag)+1)+1;
 end%%  
%%虚拟权值函数构建
J = [];
% for ik = min(W):max(W)%81个
%     II = zeros(array_num,array_num);
%    for jk = 1:array_num
%         for kk = 1:array_num
%            if array(jk) - array(kk) == ik    %%如果Pi-Pj=Pk,则II(i,j)=1/histog(k),矩阵II的其他项为0
%                 II(jk,kk) = 1/(histog(ik+abs(min(W))+1));
%             end
%         end
%     end
for ik=min_flag:max_flag
    for jk=min_flag:max_flag
        II = zeros(array_num,array_num);
        for k1=1:array_num
            for k2=1:array_num
                if array(k1,:)-array(k2,:)==[jk,ik]
                    II(k1,k2)=1/histog(ik+abs(min_flag)+1,abs(min_flag)+1-jk);
                end
            end
        end
    J = [J;reshape(II,1,[])];%%冗余阵元整合矩阵构建   %向量化II，将其作为J的第k行。
    end
end
% B=min(W):max(W);
histog_flag = round(abs(min_flag)+1);  %histog是对称的，找出中心位置
%histog_f=round(length(B)/2);
flag=histog_flag-1;
for ik = 0:min(abs(min_flag),abs(max_flag))
    if histog(histog_flag+ik) == 0 || histog(histog_flag-ik) == 0
        flag = ik-1;
        break;
    end
end  %找到连续虚拟阵元位置flag
B_vir=unique(W,'rows')*d;%虚拟阵列
B_vir(:,[1 2])=B_vir(:,[2 1]);
vir_array_line=min_flag:max_flag;
vir_array_line=vir_array_line(histog_flag-flag:histog_flag+flag);
a=[];
b=[];
for k1=1:length(vir_array_line)
    x=vir_array_line';
    a=[a;x];
    y=vir_array_line(k1)*ones(length(vir_array_line),1);
    b=[b;y];
end
vir_B(:,1)=a*d;  %连续虚拟阵列
vir_B(:,2)=b*d;
% vir_B(:,3)=zeros(length(a),1);
%协方差矩阵
R=Y*Y'/T;
z=reshape(R,[],1);
vir_Y1=J*z;
% for k=min_flag:max_flag
%     if k<vir_array_line(1)
%         vir_Y=vir_Y((max_flag-min_flag+1)*(k+abs(min_flag)+1)+1:length(vir_Y));
%     end
% end
C=reshape(vir_Y1,max_flag-min_flag+1,max_flag-min_flag+1);
C=C(histog_flag-flag:histog_flag+flag,:);
C=C(:,histog_flag-flag:histog_flag+flag);
vir_Y=reshape(C,[],1);

