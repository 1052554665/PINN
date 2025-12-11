%% MVDR (3D steering: azimuth + elevation) using RIR delays only (no manual delay)
clear; close all; clc;

%% ========== 1. Read mic positions (assume mic_positions.xlsx has X Y columns) ==========
data = readmatrix('mic_positions.xlsx');
X = data(:,1); Y = data(:,2);
Z = zeros(size(X));           % 如果你在Excel里有Z列，把这行改为 Z = data(:,3);
mic_pos = [X Y Z] / 1000;     % 假定Excel单位为 mm，直接换成 m；如果是 cm 用 /100

% center to given array center (optional)
array_center = [2.5 2 1.5];   % 房间位置（m）
mic_pos = mic_pos - mean(mic_pos,1) + array_center;     % 麦克风阵列位置

Nmic = size(mic_pos,1);
r_center = mean(mic_pos,1);   % 阵列中心（用于平面波相位参考）
fprintf('Nmic = %d, mic positions loaded (meters).\n', Nmic);


figure(1);
scatter(X, Y, 40, 'filled');
% scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 40, 'filled'); % 三维阵列图
axis equal; grid on;
xlabel('X (m)'); ylabel('Y (m)');
title('Microphone Array Geometry');

%% ========== 2. Room, RIR and signals parameters ==========
c = 340;
fs = 16000;
t = (0:1/fs:1-1/fs)';
f0 = 1000;         % target freq (Hz)
f1 = 2000;         % interference freq (Hz)
SNR = 30;          % dB
INR = 10;          % dB (interference relative to target amplitude)
nsample = 4096;    % RIR length

% source 3D positions
s_target = [2 3.5 4];   % used for RIR generation (m)
s_interf  = [2.5 3.8 4];

%% ====================== 可视化房间、声源和麦克风阵列位置 ======================
figure(2); clf;
hold on; grid on; axis equal;

% 1. 绘制房间边界（矩形框）
L = [5 4 6];  % 房间长宽高
xv = [0 L(1) L(1) 0 0];
yv = [0 0 L(2) L(2) 0];
z0 = zeros(size(xv));

% 地面和顶面
plot3(xv, yv, z0, 'k--', 'LineWidth', 1.2);
plot3(xv, yv, L(3)*ones(size(z0)), 'k--', 'LineWidth', 1.2);

% 垂直边
for i = 1:4
    plot3([xv(i) xv(i)], [yv(i) yv(i)], [0 L(3)], 'k--');
end

% 2. 绘制麦克风阵列位置
scatter3(mic_pos(:,1), mic_pos(:,2), mic_pos(:,3), 60, 'filled', 'b');
text(mean(mic_pos(:,1)), mean(mic_pos(:,2)), mean(mic_pos(:,3))+0.1, 'Mic Array', ...
     'Color', 'b', 'FontWeight', 'bold');

% 3. 绘制目标声源与干扰源
scatter3(s_target(1), s_target(2), s_target(3), 100, 'r', 'filled');
text(s_target(1), s_target(2), s_target(3)+0.1, 'Target Source', 'Color', 'r', 'FontWeight', 'bold');

scatter3(s_interf(1), s_interf(2), s_interf(3), 100, 'm', 'filled');
text(s_interf(1), s_interf(2), s_interf(3)+0.1, 'Interference', 'Color', 'm', 'FontWeight', 'bold');

% 4. 可选：连接阵列中心与声源方向
r_center = mean(mic_pos,1);
plot3([r_center(1) s_target(1)], [r_center(2) s_target(2)], [r_center(3) s_target(3)], 'r--', 'LineWidth', 1.2);
plot3([r_center(1) s_interf(1)], [r_center(2) s_interf(2)], [r_center(3) s_interf(3)], 'm--', 'LineWidth', 1.2);

% 5. 设置视角与标签
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Room, Sources, and Microphone Array Layout');
view(45, 25);
legend({'Room boundary','Microphones','Target Source','Interference'}, 'Location','best');

%% ========== 3. Fraunhofer check (far-field criterion) ==========
D = 0.15;           % 阵列最大孔径 (m) — 150 mm
f_max = 5000;       % 关心的最高频率（Hz）
lambda_min = c / f_max;
R_far = 2 * D^2 / lambda_min;   % Fraunhofer distance (approx)

fprintf('Array diameter D=%.3f m, f_max=%d Hz, lambda_min=%.3f m\n', D, f_max, lambda_min);
fprintf('Fraunhofer distance R_far ≈ %.3f m\n', R_far);

% compute actual distances of sources to array center
dist_s_target = norm(s_target - r_center);
dist_s_interf  = norm(s_interf  - r_center);

fprintf('Distance target->array center = %.3f m, interference = %.3f m\n', dist_s_target, dist_s_interf);

use_farfield_target = dist_s_target >= R_far;
use_farfield_interf  = dist_s_interf  >= R_far;

if ~use_farfield_target
    fprintf('Warning: target is within near-field (%.3f < %.3f). Using near-field steering for target.\n', dist_s_target, R_far);
else
    fprintf('Target satisfies far-field criterion. Using plane-wave steering for target.\n');
end
if ~use_farfield_interf
    fprintf('Warning: interference is within near-field. Using near-field steering for interference.\n');
else
    fprintf('Interference satisfies far-field criterion. Using plane-wave steering for interference.\n');
end

%% ========== 4. Generate RIRs (per-mic) ==========
L = [5 4 6];    % room dims
beta = 0;     % wall reflection coefficient
mtype = 'omnidirectional';  % depends on your rir_generator
order = -1;
dim = 3;
orientation = 0;
hp_filter = 1;

h_target = zeros(Nmic, nsample);    % 初始化目标和干扰的RIR矩阵
h_interf = zeros(Nmic, nsample);

fprintf('Generating RIRs for %d microphones (this may take a while)...\n', Nmic);
for m = 1:Nmic      % 循环为每个麦克风生成目标和干扰的RIR，并确保为行向量后存储。
    rm = mic_pos(m,:);
    ht = rir_generator(c, fs, rm, s_target, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    % ensure row vectors
    if iscolumn(ht), ht = ht.'; end
    if iscolumn(hi), hi = hi.'; end
    h_target(m,:) = ht;
    h_interf(m,:)  = hi;
end
fprintf('RIR generation done.\n');

%% ========== 5. Generate source signals and convolve with RIR (do NOT add manual delays) ==========
x_target = sin(2*pi*f0*t);
x_interf  = sin(2*pi*f1*t);

Nt = length(x_target);  % 初始化信号矩阵
X_target = zeros(Nt, Nmic);
X_interf  = zeros(Nt, Nmic);

% convolution (same length)
for m = 1:Nmic      % 每个麦克风通道进行卷积，保持相同长度
    X_target(:,m) = conv(x_target, h_target(m,:), 'same');
    X_interf(:,m)  = conv(x_interf,  h_interf(m,:),  'same');
end

% scale interference by INR (dB) 根据INR缩放干扰信号
X_interf = X_interf / (10^(INR/20));    

% add noise to meet overall SNR per channel (relative to target)
% 计算目标信号的RMS，根据SNR计算期望的噪声RMS，生成高斯噪声并调整其RMS
sig_rms = sqrt(mean(X_target.^2, 1));
desired_noise_rms = sig_rms ./ (10^(SNR/20));
noise = randn(size(X_target));
cur_noise_rms = sqrt(mean(noise.^2, 1));
noise = noise .* (desired_noise_rms ./ cur_noise_rms);

X_noisy = X_target + X_interf + noise;  % 合成带噪信号
fprintf('Signals prepared (RIR only). No extra delays added.\n');

%% ========== 6. STFT parameters and compute STFT per channel ==========
Nfft = 1024;
win = hamming(512);
noverlap = 256;

% compute STFT channel-wise; we store as S(freq, time, channel)
% 对每个麦克风通道进行STFT，结果存储在S（频率×时间×通道）
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
Fbins = length(F);  % 获取频率点和时间帧
Tframes = length(T);

% limit up to f_limit (0-5kHz)
f_limit = find(F <= 5000, 1, 'last');   % 找到5000Hz以下的频率索引
fprintf('STFT computed: %d freq bins, %d time frames. Using bins 1..%d up to %.1f Hz.\n', Fbins, Tframes, f_limit, F(f_limit));

%% ========== 7. 修正导向矢量构建 ==========
a_target = zeros(Nmic, Fbins);
a_interf = zeros(Nmic, Fbins);

% 使用实际位置计算方向向量（而不是手动设置的az/el）
vec_target_actual = s_target - r_center;
vec_interf_actual = s_interf - r_center;

% 计算实际的方向角和仰角
az_target_actual = atan2d(vec_target_actual(2), vec_target_actual(1));
el_target_actual = asind(vec_target_actual(3) / norm(vec_target_actual));
az_interf_actual = atan2d(vec_interf_actual(2), vec_interf_actual(1));  
el_interf_actual = asind(vec_interf_actual(3) / norm(vec_interf_actual));

fprintf('修正后的目标方向: az=%.1f°, el=%.1f°\n', az_target_actual, el_target_actual);
fprintf('修正后的干扰方向: az=%.1f°, el=%.1f°\n', az_interf_actual, el_interf_actual);

% 更新全局变量，确保一致性
az_target = az_target_actual;
el_target = el_target_actual;  
az_interf = az_interf_actual;
el_interf = el_interf_actual;

for k = 1:Fbins
    freq = F(k);
    if k > f_limit
        a_target(:,k) = zeros(Nmic,1);
        a_interf(:,k) = zeros(Nmic,1);
        continue;
    end

    % 统一使用修正后的方向计算
    d_target_corrected = [cosd(el_target)*cosd(az_target);
                         cosd(el_target)*sind(az_target); 
                         sind(el_target)];
    d_interf_corrected = [cosd(el_interf)*cosd(az_interf);
                         cosd(el_interf)*sind(az_interf);
                         sind(el_interf)];

    if use_farfield_target
        tau_proj = (mic_pos - r_center) * d_target_corrected;
        a_target(:,k) = exp(-1j*2*pi*freq*tau_proj / c);
    else
        for m = 1:Nmic
            dist = norm(mic_pos(m,:) - s_target);
            a_target(m,k) = exp(-1j*2*pi*freq*dist/c) / max(dist, 1e-6);
        end
    end

    if use_farfield_interf
        tau_proj_i = (mic_pos - r_center) * d_interf_corrected;
        a_interf(:,k) = exp(-1j*2*pi*freq*tau_proj_i / c);
    else
        for m = 1:Nmic
            disti = norm(mic_pos(m,:) - s_interf);
            a_interf(m,k) = exp(-1j*2*pi*freq*disti/c) / max(disti, 1e-6);
        end
    end

    % 归一化
    if norm(a_target(:,k)) > 0
        a_target(:,k) = a_target(:,k) / norm(a_target(:,k));
    end
    if norm(a_interf(:,k)) > 0  
        a_interf(:,k) = a_interf(:,k) / norm(a_interf(:,k));
    end
end

%% ========== 8. Memory-safe per-frame MVDR (diagonal loading adapted) ==========
fprintf('Running MVDR (per-frequency, per-frame) ...\n');
Yf = zeros(Fbins, Tframes);   % complex freq x time result

% Tunable parameters (you can try Mavg=21/epsilon=1e-2, or smaller)
Mavg = 21;                     % averaging window for covariance (frames)
epsilon = 1e-2;                % base loading (relative scale)

% optional shrinkage factor for additional stability (set 0 to disable)
shrink_alpha = 0.05;           % 0..0.3 typical; 0 means no shrinkage

tic
for k = 1:f_limit       % 循环每个频率，获取该频率下的多通道数据（Nmic×Tframes），并检查导向矢量是否为零
    Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes
    at = a_target(:,k);
    if norm(at)==0
        continue;
    end

    for tt = 1:Tframes      % 对于每一帧，取前后共Mavg帧数据，计算协方差矩阵
        t1 = max(1, tt - floor(Mavg/2));
        t2 = min(Tframes, tt + floor(Mavg/2));
        Xloc = Xkf(:, t1:t2);
        Rxx = (Xloc * Xloc') / size(Xloc,2);

        % shrinkage toward scaled identity  如果使用收缩，则向缩放单位矩阵收缩
        if shrink_alpha > 0
            R_shrink = (1-shrink_alpha) * Rxx + shrink_alpha * (trace(Rxx)/Nmic) * eye(Nmic);
            Rxx = R_shrink;
        end

        % adaptive diagonal loading proportional to trace   自适应对角加载
        reg = epsilon * trace(Rxx) / Nmic;
        Rxx = Rxx + reg * eye(Nmic);

        % MVDR weight: w = R^{-1} a / (a^H R^{-1} a) 计算MVDR权重
        w = Rxx \ at;
        denom = (at' * w);
        if abs(denom) < 1e-12
            W = zeros(size(w));
        else
            W = w ./ denom;
        end

        % output for frame tt   计算输出STFT
        Yf(k, tt) = W' * Xkf(:, tt);
    end
end
toc

%% ========== Diagnostic for 2 kHz and Rxx (place right after MVDR loop, before ISTFT) ==========
% find freq bin for 2kHz and 1kHz
[~, k2] = min(abs(F - 2000));
[~, k1] = min(abs(F - 1000));

% pick a robust frame interval (middle 30% of frames) for Rxx estimate
numFrames = size(Yf,2);
t1_diag = max(1, round(numFrames*0.35));
t2_diag = min(numFrames, round(numFrames*0.65));
if t1_diag >= t2_diag
    t1_diag = 1; t2_diag = numFrames;
end

% get Xkf for k2 (freq x time x channel earlier)
Xkf_k2 = squeeze(S(k2,:,:)).';   % Nmic x Tframes
% safe bounds
t2_diag = min(t2_diag, size(Xkf_k2,2));
Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2);
reg_diag = epsilon * trace(Rxx_diag) / Nmic;
Rxx_diag = Rxx_diag + reg_diag * eye(Nmic);

% eigenspectrum
[~, D] = eig(Rxx_diag);
evals = sort(diag(D), 'descend');
figure(3); plot(1:Nmic, 10*log10(evals + eps), '-o');
xlabel('Index'); ylabel('Eigenvalue (dB)'); title('Rxx eig-spectrum at ~2 kHz');

% compute MVDR weight used at k2 for the center frame
center_frame = round(numFrames/2);
t1c = max(1, center_frame - floor(Mavg/2));
t2c = min(numFrames, center_frame + floor(Mavg/2));
Xlocc = Xkf_k2(:, t1c:t2c);
Rxxc = (Xlocc * Xlocc') / size(Xlocc,2) + (epsilon * trace(Xlocc * Xlocc') / Nmic) * eye(Nmic);

at_k2 = a_target(:, k2);
w_diag = Rxxc \ at_k2;
denom_diag = (at_k2' * w_diag);
if abs(denom_diag) < 1e-12
    Wdiag = zeros(size(w_diag));
else
    Wdiag = w_diag ./ denom_diag;
end

% beampattern at 2 kHz over azimuth (elev=0)
azs = -180:1:180;
resp = zeros(size(azs));
for ii = 1:length(azs)
    d_try = [cosd(0)*cosd(azs(ii)); cosd(0)*sind(azs(ii)); sind(0)];
    a_try = exp(-1j*2*pi*F(k2) * ((mic_pos - r_center) * d_try) / c);
    a_try = a_try / norm(a_try);
    resp(ii) = 20*log10(abs(Wdiag' * a_try) + eps);
end
figure(4); plot(azs, resp); grid on;
xlabel('Azimuth (deg)'); ylabel('Response (dB)'); title(sprintf('Beampattern at %.1f Hz', F(k2)));

fprintf('Diagnostic done. See eig-spectrum and beampattern for 2 kHz.\n');

%% ========== 9. ISTFT and normalization ==========
y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
% istft may return a slightly different length; trim/pad to original length
y_mvdr = real(y_mvdr);

% robust length alignment   调整输出长度与输入相同
if length(y_mvdr) >= Nt
    y_mvdr = y_mvdr(1:Nt);
else
    y_mvdr = [y_mvdr; zeros(Nt - length(y_mvdr), 1)];
end

% energy normalization (avoid clipping)
y_mvdr = y_mvdr / max(abs(y_mvdr) + 1e-12);

%% ========== 10. Diagnostics: show waveforms & spectrograms & PSD around 1kHz ==========
figure(5); clf;
subplot(3,1,1); plot((0:Nt-1)/fs, X_target(:,1)); title('Target (mic 1, with RIR)'); xlabel('Time (s)');
subplot(3,1,2); plot((0:Nt-1)/fs, X_noisy(:,1));  title('Noisy (mic 1)'); xlabel('Time (s)');
subplot(3,1,3); plot((0:Nt-1)/fs, y_mvdr);         title('MVDR output (time)'); xlabel('Time (s)');

figure(6); clf;
subplot(3,1,1); spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis'); title('Target spectrogram (mic1)');
subplot(3,1,2); spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis'); title('Noisy spectrogram (mic1)');
subplot(3,1,3); spectrogram(y_mvdr,         hamming(256), 128, 512, fs, 'yaxis'); title('MVDR output spectrogram');

% PSD around 1 kHz
nfft_psd = 4096;
[Pxx_in, Fp]  = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, fs);
[Pxx_out, ~]  = pwelch(y_mvdr,        hann(1024), 512, nfft_psd, fs);

figure(7); clf;
plot(Fp, 10*log10(Pxx_in + eps)); hold on;
plot(Fp, 10*log10(Pxx_out + eps), 'LineWidth', 1.5);
xlim([800 1200]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Mic1 noisy','MVDR output'}); title('PSD around 1 kHz');

% Optional: PSD around 2 kHz for direct inspection
figure(8); clf;
plot(Fp, 10*log10(Pxx_in + eps)); hold on;
plot(Fp, 10*log10(Pxx_out + eps), 'LineWidth', 1.5);
xlim([1800 2200]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Mic1 noisy','MVDR output'}); title('PSD around 2 kHz');

%% ========== 11. Save output audio ==========
audiowrite('y_mvdr.wav', y_mvdr, fs);
fprintf('Saved MVDR output to y_mvdr.wav\n');

%% ========== End ==========
fprintf('Processing complete. Check spectrogram, PSD and diagnostics: 1 kHz should be visible/enhanced; 2 kHz beampattern/eig-spectrum available for tuning.\n');
