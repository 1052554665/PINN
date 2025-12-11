function [X,gamma_ind,gamma_est,count] = MSBL(Phi, Y, lambda, Learn_Lambda, varargin)
[N M] = size(Phi); 
[N L] = size(Y);  
% Default Control Parameters 默认控制参数
PRUNE_GAMMA = 1e-4;       % threshold for prunning small hyperparameters gamma_i    参数gamma减小的阈值
EPSILON     = 1e-6;       % threshold for stopping iteration.  停止迭代的阈值
MAX_ITERS   = 7000;       % maximum iterations  最大迭代次数
PRINT       = 1;          % don't show progress information  不显示进度信息

if(mod(length(varargin),2)==1)    %mod（23，5）=3标量被除后的余数（取模运算）
    error('Optional parameters should always go by pairs\n');   %可选参数应始终成对排列
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'prune_gamma'
                PRUNE_GAMMA = varargin{i+1}; 
            case 'epsilon'   
                EPSILON = varargin{i+1}; 
            case 'print'    
                PRINT = varargin{i+1}; 
            case 'max_iters'
                MAX_ITERS = varargin{i+1};  
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end

if (PRINT) fprintf('\nRunning MSBL ...\n'); end  %如果print=1, 输出nRunning MSBL...


% Initializations  初始化
gamma = ones(M,1); 
keep_list = [1:M]';
m = length(keep_list);
mu = zeros(M,L);
count = 0;                        % iteration count  迭代计数


% *** Learning loop *** 学习循环
while (1)

    % *** Prune weights as their hyperparameters go to zero ***减小权重，因为超参数为0
    if (min(gamma) < PRUNE_GAMMA )
        index = find(gamma > PRUNE_GAMMA);
        gamma = gamma(index);  % use all the elements larger than MIN_GAMMA to form new 'gamma' 使用所有大于min_gamma的元素形成新的gamma
        Phi = Phi(:,index);    % corresponding columns in Phi  Phi中的对应列
        keep_list = keep_list(index);
        m = length(gamma);
    end;


    mu_old =mu;
    Gamma = diag(gamma);
    G = diag(sqrt(gamma));
        
    % ****** estimate the solution matrix *****估计解决方案矩阵
%     [U,S,V] = svd(Phi*G,'econ');
%    
%     [d1,d2] = size(S);
%     if (d1 > 1)     diag_S = diag(S);
%     else            diag_S = S(1);      end;
%        
%     Xi = G * V * diag((diag_S./(diag_S.^2 + lambda + 1e-16))) * U';
%     mu = Xi * Y;
%计算mu
I=eye(N);
Sigma_t=lambda*I+Phi*Gamma*Phi';
Sigma_t1=inv(Sigma_t);
mu=Gamma*Phi'*Sigma_t1*Y;
 % *** Update hyperparameters, i.e. Eq(18) in the reference *** %更新超参数
    gamma_old = gamma;
    mu2_bar = sum(abs(mu).^2,2)/L;
%    Sigma_w_diag = real( gamma - (sum(Xi'.*(Phi*Gamma)))');

% Sigma_w=Gamma-Gamma*Phi'*Sigma_t1*Phi*Gamma;
% Sigma_w_diag=diag(Sigma_w);
Xi=Gamma*Phi'*Sigma_t1*Phi*Gamma;
Sigma_i_diag=diag(Xi);
Sigma_w_diag=gamma-Sigma_i_diag;
gamma = mu2_bar + Sigma_w_diag;
    % ***** the lambda learning rule *****
    % You can use it to estimate the lambda when SNR >= 20 dB. But when SNR < 20 dB, 
    % you'd better use other methods to estimate the lambda, since it is not robust 
    % in strongly noisy cases (in simulations, you can feed it with the
    % true noise variance, which can lead to a near-optimal performance)
    if Learn_Lambda == 1
        lambda = (norm(Y - Phi * mu,'fro')^2/L)/(N-m + sum(Sigma_w_diag./gamma_old));   
    end;
    
    
    % *** Check stopping conditions, etc. ***
    count = count + 1;
    if (PRINT) disp(['iters: ',num2str(count),'   num coeffs: ',num2str(m), ...
            '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
    if (count >= MAX_ITERS) break;  end;

    if (size(mu) == size(mu_old))
        dmu = max(max(abs(mu_old - mu)));
        if (dmu < EPSILON)  break;  end;
    end;

end;


% Expand hyperparameters 扩展超参数
gamma_ind = sort(keep_list);
gamma_est = zeros(M,1);
gamma_est(keep_list,1) = gamma;  

% expand the final solution
X = zeros(M,L);
X(keep_list,:) = mu; 

if (PRINT) fprintf('\nFinish running ...\n'); end
return;