% 将目标信号和干扰分别替换为读取真实的信号，MVDR时频图去掉横纵坐标
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


%% ========== 2. Room, RIR and signals parameters ==========
c = 340;
fs = 44100;            % 统一采样率 (Hz) —— RIR 与 STFT 使用此采样率
SNR = 30;              % dB (噪声相对于目标)
INR = 10;              % dB (干扰相对于目标) -- 脚本会把干扰缩放为 RMS_interf = RMS_target * 10^(INR/20)
nsample = 4096;        % RIR length

% source 3D positions (用于 RIR 生成)
s_target = [2 3.5 4];   % m
s_interf  = [2.5 3.8 4];


%% ========== 3. Fraunhofer check (far-field criterion) ==========
D = 0.15;           % 阵列最大孔径 (m) — 150 mm
f_max = 5000;       % 关心的最高频率（Hz）
lambda_min = c / f_max;
R_far = 2 * D^2 / lambda_min;   % Fraunhofer distance (approx)

fprintf('Array diameter D=%.3f m, f_max=%d Hz, lambda_min=%.3f m\n', D, f_max, lambda_min);
fprintf('Fraunhofer distance R_far ≈ %.3f m\n', R_far);

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
L = [5 4 6];    % room dims (m)
beta = 0;       % wall reflection coefficient
mtype = 'omnidirectional';
order = -1;
dim = 3;
orientation = 0;
hp_filter = true;

h_target = zeros(Nmic, nsample);    % 初始化目标和干扰的RIR矩阵
h_interf = zeros(Nmic, nsample);

fprintf('Generating RIRs for %d microphones (this may take a while)...\n', Nmic);
for m = 1:Nmic
    rm = mic_pos(m,:);
    ht = rir_generator(c, fs, rm, s_target, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    hi = rir_generator(c, fs, rm, s_interf, L, beta, nsample, mtype, order, dim, orientation, hp_filter);
    if iscolumn(ht), ht = ht.'; end
    if iscolumn(hi), hi = hi.'; end
    h_target(m,:) = ht;
    h_interf(m,:)  = hi;
end
fprintf('RIR generation done.\n');


%% ========== 5. Read real signals (replace synthetic sinusoids) ==========
% 要求：在工作目录下准备好 target.wav 与 interf.wav（任意采样率、单/多通道均可）
[target_src, fs_t] = audioread('G4_10pSeventhHarmonic_1.wav');
[interf_src, fs_i]  = audioread('振安1#反_part46.wav');

% convert to mono if needed
if size(target_src,2) > 1
    target_src = mean(target_src, 2);
end
if size(interf_src,2) > 1
    interf_src = mean(interf_src, 2);
end

% resample to common fs if needed
if fs_t ~= fs
    target_src = resample(target_src, fs, fs_t);
    fprintf('Resampled target.wav from %d Hz to %d Hz\n', fs_t, fs);
end
if fs_i ~= fs
    interf_src = resample(interf_src, fs, fs_i);
    fprintf('Resampled interf.wav from %d Hz to %d Hz\n', fs_i, fs);
end

% normalize (避免过大幅值) —— 你可以根据需要调整
target_src = target_src / (max(abs(target_src)) + 1e-12);
interf_src  = interf_src  / (max(abs(interf_src))  + 1e-12);

% align lengths: pad shorter with zeros so两信号具有相同长度
len_t = length(target_src);
len_i = length(interf_src);
Nt_src = max(len_t, len_i);
target_src = [target_src; zeros(Nt_src - len_t, 1)];
interf_src = [interf_src; zeros(Nt_src - len_i, 1)];

% scale interference according to INR (dB): target RMS * 10^(INR/20)
rms_t = sqrt(mean(target_src.^2));
rms_i = sqrt(mean(interf_src.^2));
scale_factor = (rms_t * 10^(INR/20)) / (rms_i + eps);
interf_src = interf_src * scale_factor;
fprintf('Real signals loaded. target length=%d samp, interf length=%d samp (padded/truncated to %d). INR scale factor=%.3f\n', len_t, len_i, Nt_src, scale_factor);

% set Nt (用于后续处理)
Nt = Nt_src;


%% ========== 6. Convolve source signals with RIR for each mic (same length 'same') ==========
X_target = zeros(Nt, Nmic);
X_interf  = zeros(Nt, Nmic);

for m = 1:Nmic
    xt = conv(target_src, h_target(m,:), 'same');   % same 长度 Nt
    xi = conv(interf_src,  h_interf(m,:),  'same');
    X_target(:,m) = xt;
    X_interf(:,m)  = xi;
end

% add noise to meet overall SNR per channel (relative to target)
sig_rms = sqrt(mean(X_target.^2, 1));                       % 每通道目标 RMS
desired_noise_rms = sig_rms ./ (10^(SNR/20));
noise = randn(size(X_target));
cur_noise_rms = sqrt(mean(noise.^2, 1));
noise = noise .* (desired_noise_rms ./ (cur_noise_rms + 1e-12));

X_noisy = X_target + X_interf + noise;  % 合成带噪信号
fprintf('Signals prepared (RIR-convolved real signals). Added noise for SNR=%d dB.\n', SNR);


%% ========== 7. STFT parameters and compute STFT per channel ==========
Nfft = 1024;
win = hamming(512);
noverlap = 256;

for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
Fbins = length(F);
Tframes = length(T);

% limit up to f_limit (0-5kHz)
f_limit = find(F <= 5000, 1, 'last');
fprintf('STFT computed: %d freq bins, %d time frames. Using bins 1..%d up to %.1f Hz.\n', Fbins, Tframes, f_limit, F(f_limit));


%% ========== 8. Steering vectors based on actual source positions ==========
a_target = zeros(Nmic, Fbins);
a_interf = zeros(Nmic, Fbins);

for k = 1:Fbins
    freq = F(k);
    if k > f_limit
        a_target(:,k) = zeros(Nmic,1);
        a_interf(:,k) = zeros(Nmic,1);
        continue;
    end

    for m = 1:Nmic
        dist_target = norm(mic_pos(m,:) - s_target);
        a_target(m,k) = exp(-1j*2*pi*freq*dist_target/c) / max(dist_target, 1e-6);
        dist_interf = norm(mic_pos(m,:) - s_interf);
        a_interf(m,k) = exp(-1j*2*pi*freq*dist_interf/c) / max(dist_interf, 1e-6);
    end

    if norm(a_target(:,k)) > 0
        a_target(:,k) = a_target(:,k) / norm(a_target(:,k));
    end
    if norm(a_interf(:,k)) > 0
        a_interf(:,k) = a_interf(:,k) / norm(a_interf(:,k));
    end
end

fprintf('使用基于实际声源位置的导向矢量构建（real signals）\n');


%% ========== 9. Memory-safe per-frame MVDR (diagonal loading adapted) ==========
fprintf('Running MVDR (per-frequency, per-frame) ...\n');
Yf = zeros(Fbins, Tframes);   % complex freq x time result

Mavg = 21;                     % averaging window for covariance (frames)
epsilon = 1e-2;                % base loading
shrink_alpha = 0.05;           % shrinkage factor

tic
for k = 1:f_limit
    Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes
    at = a_target(:,k);
    if norm(at)==0
        continue;
    end

    for tt = 1:Tframes
        t1 = max(1, tt - floor(Mavg/2));
        t2 = min(Tframes, tt + floor(Mavg/2));
        Xloc = Xkf(:, t1:t2);
        Rxx = (Xloc * Xloc') / size(Xloc,2);

        if shrink_alpha > 0
            R_shrink = (1-shrink_alpha) * Rxx + shrink_alpha * (trace(Rxx)/Nmic) * eye(Nmic);
            Rxx = R_shrink;
        end

        reg = epsilon * trace(Rxx) / Nmic;
        Rxx = Rxx + reg * eye(Nmic);

        w = Rxx \ at;
        denom = (at' * w);
        if abs(denom) < 1e-12
            W = zeros(size(w));
        else
            W = w ./ denom;
        end

        Yf(k, tt) = W' * Xkf(:, tt);
    end
end
toc


%% ========== 10. Diagnostics for 2 kHz and Rxx ==========
[~, k2] = min(abs(F - 2000));
[~, k1] = min(abs(F - 1000));
numFrames = size(Yf,2);
t1_diag = max(1, round(numFrames*0.35));
t2_diag = min(numFrames, round(numFrames*0.65));
if t1_diag >= t2_diag
    t1_diag = 1; t2_diag = numFrames;
end

Xkf_k2 = squeeze(S(k2,:,:)).';
t2_diag = min(t2_diag, size(Xkf_k2,2));
Xloc_diag = Xkf_k2(:, t1_diag:t2_diag);
Rxx_diag = (Xloc_diag * Xloc_diag') / size(Xloc_diag,2);
reg_diag = epsilon * trace(Rxx_diag) / Nmic;
Rxx_diag = Rxx_diag + reg_diag * eye(Nmic);

[~, D] = eig(Rxx_diag);
evals = sort(diag(D), 'descend');

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

azs = -180:1:180;
resp = zeros(size(azs));
for ii = 1:length(azs)
    d_try = [cosd(0)*cosd(azs(ii)); cosd(0)*sind(azs(ii)); sind(0)];
    a_try = exp(-1j*2*pi*F(k2) * ((mic_pos - r_center) * d_try) / c);
    a_try = a_try / norm(a_try);
    resp(ii) = 20*log10(abs(Wdiag' * a_try) + eps);
end
fprintf('Diagnostic done. See eig-spectrum and beampattern for 2 kHz.\n');


%% ========== 11. ISTFT and normalization ==========
y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
y_mvdr = real(y_mvdr);

if length(y_mvdr) >= Nt
    y_mvdr = y_mvdr(1:Nt);
else
    y_mvdr = [y_mvdr; zeros(Nt - length(y_mvdr), 1)];
end

y_mvdr = y_mvdr / (max(abs(y_mvdr)) + 1e-12);


%% ========== 12. Diagnostics: show waveforms & spectrograms & PSD around 1kHz ==========
figure; clf;
spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
         
set(gcf, 'Units','pixels', 'Position',[100 100 224 224]);
set(gca,'Units','normalized','Position',[0 0 1 1]);
axis off;   % 完全关闭坐标轴


%% ========== 13. Save output audio ==========
audiowrite('y_mvdr.wav', y_mvdr, fs);
fprintf('Saved MVDR output to y_mvdr.wav\n');

fprintf('Processing complete. Check spectrogram, PSD and diagnostics.\n');
