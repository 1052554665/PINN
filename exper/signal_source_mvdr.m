%% 真实场景：单声源，变压器，MVDR

% 完整流程：读取多通道 -> MUSIC DOA 估计（远场）-> 构建 steering vector -> 频域 MVDR
% 适用于单目标场景
clear; close all; clc;c

%% -------------------- 用户参数（可修改） --------------------
c = 340;                 % 声速 (m/s)
f_max = 5000;            % 处理上限频率 (Hz)
music_scan_res = 1;      % MUSIC 扫描角度分辨率 (deg)
num_music_freqs = 5;     % 用于 IMUSIC 的频率个数（在低中频选取）
alpha_load = 1e-3;       % MVDR 对角加载系数
vad_energy_factor = 0.5; % VAD 阈值因子（参考通道能量中位数 * factor）
% STFT 参数
Nfft = 1024;
win = hamming(512);
noverlap = 256;

%% -------------------- 读取 mic_pos（必须） --------------------
% 期待 mic_pos 为 Nmic x 3（x,y,z）矩阵（单位 m）。如果没有，请读取文件或手动定义。
if exist('mic_pos','var') ~= 1
    % 尝试从文件 mic_positions.xlsx 读取（如果存在）
    if exist('mic_positions.xlsx','file')
        data = readmatrix('mic_positions.xlsx');
        X = data(:,1); Y = data(:,2); Z = zeros(size(X));
        % 若 Excel 单位为 mm 或 cm 请在此行转换（示例假设 mm）
        mic_pos = [X Y Z] / 1000;  % 修改为适合你的单位转换
    else
        error('请在运行脚本前定义 mic_pos (Nmic x 3) 或放置 mic_positions.xlsx 文件.');
    end
end
Nmic = size(mic_pos,1);

%% -------------------- 读取多通道录音 --------------------
% 支持四个 32 通道 wav 文件自动拼接为 128 通道
wav_files = {
    'record CH1~CH32.wav';
    'record CH33~CH64.wav';
    'record CH65~CH96.wav';
    'record CH97~CH128.wav'
};
exist_flags = cellfun(@(f) exist(f,'file'), wav_files);
if all(exist_flags)
    fprintf('检测到 4 个多通道文件，开始拼接为 128 通道信号...\n');
    X_parts = cell(4,1);
    for i = 1:4
        [data, fs] = audioread(wav_files{i});
        X_parts{i} = data;
        fprintf('  %s 读取完成 (%d 通道)\n', wav_files{i}, size(data,2));
    end
    % 检查长度一致
    Lmin = min(cellfun(@(x) size(x,1), X_parts));
    for i = 1:4
        X_parts{i} = X_parts{i}(1:Lmin,:);
    end
    % 拼接通道维度
    X_noisy = [X_parts{1}, X_parts{2}, X_parts{3}, X_parts{4}];
    Nmic = size(X_noisy,2);
    fprintf('共 %d 通道，采样率 %d Hz，样本数 %d。\n', Nmic, fs, size(X_noisy,1));
else
    error('未检测到全部四个分段 wav 文件，请检查文件名。');
end

Nsamples = size(X_noisy,1);
t = (0:Nsamples-1)'/fs;

%% -------------------- 计算 STFT（每通道） --------------------
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
Fbins = length(F);
Tframes = length(T);
f_limit = find(F <= f_max, 1, 'last');

%% -------------------- 用 IMUSIC 做 wideband MUSIC（估计 DOA） --------------------
f_low = 300;
freqs_idx = round(linspace(find(F>=f_low,1), f_limit, ...
    min(num_music_freqs, f_limit - find(F>=f_low,1)+1)));
if isempty(freqs_idx)
    error('MUSIC 频点选择失败，请检查参数。');
end
 
angles = 0:music_scan_res:359;
Pims = zeros(length(angles), length(freqs_idx));
 
for fi = 1:length(freqs_idx)
    k = freqs_idx(fi);
    Xk = squeeze(S(k,:,:)).';
    Rxx = (Xk * Xk') / Tframes;
    [V, D] = eig(Rxx);
    [dvals, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    En = V(:, 2:end);
    for aidx = 1:length(angles)
        theta = angles(aidx);
        u = [cosd(theta); sind(theta); 0];
        freq = F(k);
        a = exp(-1j * 2*pi * freq * (mic_pos * u) / c);
        a = a / norm(a);
        Pims(aidx, fi) = 1 / (a' * (En * En') * a + eps);
    end
end
Pavg = mean(abs(Pims), 2);
[~, sortidx] = sort(Pavg, 'descend');
best_angle = angles(sortidx(1));
fprintf('Estimated DOA (azimuth) by IMUSIC: %.2f deg\n', best_angle);

%% -------------------- 由 DOA 计算 TDOA --------------------
r_ref = mean(mic_pos,1);
u_hat = [cosd(best_angle); sind(best_angle); 0];
tau_m = -((mic_pos - r_ref) * u_hat) / c;
fprintf('TDOA relative to array center (first 5 channels):\n');
disp(tau_m(1:min(5,end)));

%% -------------------- 构建 steering vector（用于 MVDR） --------------------
a_f = zeros(Nmic, Fbins);
for k = 1:Fbins
    freq = F(k);
    a_f(:,k) = exp(-1j * 2*pi * freq * (mic_pos * u_hat) / c);
    a_f(:,k) = a_f(:,k) ./ norm(a_f(:,k));
end

%% -------------------- 频域 MVDR --------------------
refChan = 1;
energy_ref = squeeze(sum(abs(S(:,:,refChan)).^2,1));
vad_mask = energy_ref > median(energy_ref) * vad_energy_factor;

Yf = zeros(Fbins, Tframes);
for k = 1:f_limit
    Xkf = squeeze(S(k,:,:)).';
    noise_idx = find(~vad_mask);
    if isempty(noise_idx)
        Rnn = (Xkf * Xkf')/Tframes;
    else
        Xn = Xkf(:, noise_idx);
        Rnn = (Xn * Xn') / size(Xn,2);
    end
    Rnn = Rnn + alpha_load*(trace(Rnn)/Nmic)*eye(Nmic);
    a_k = a_f(:,k);
    w = (Rnn \ a_k) / (a_k' * (Rnn \ a_k) + eps);
    Yf(k,:) = w' * Xkf;
end

y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
win_gain = sum(win.^2) / (length(win) - noverlap);
y_mvdr = real(y_mvdr) / win_gain;

%% -------------------- 可视化与保存 --------------------
figure('Name','DOA & Signals','NumberTitle','off');
subplot(3,1,1); plot(t, X_noisy(:,1)); title('Reference Channel (raw)');
subplot(3,1,2); plot(t, y_mvdr); title('MVDR Output');
subplot(3,1,3);
plot(angles, 10*log10(Pavg/max(Pavg))); grid on;
xlabel('Azimuth (deg)'); ylabel('Normalized MUSIC (dB)');
title(sprintf('IMUSIC Spectrum — Peak at %.2f°', best_angle));

figure('Name','Spectrograms','NumberTitle','off');
subplot(2,1,1);
spectrogram(X_noisy(:,refChan), hamming(256), 128, 512, fs, 'yaxis');
title('Reference Channel Spectrogram');
subplot(2,1,2);
spectrogram(y_mvdr, hamming(256), 128, 512, fs, 'yaxis');
title('MVDR Output Spectrogram');

audiowrite('mvdr_output.wav', y_mvdr, fs);
fprintf('MVDR 输出已保存为 mvdr_output.wav\n');

%% -------------------- 辅助函数 --------------------
function tau = gcc_phat_tdoa(sig1, sig2, fs)
    N = length(sig1);
    nfft = 2^nextpow2(2*N-1);
    S1 = fft(sig1, nfft);
    S2 = fft(sig2, nfft);
    R = (S1 .* conj(S2)) ./ (abs(S1 .* conj(S2)) + eps);
    cc = real(ifft(R));
    cc = fftshift(cc);
    lags = (-nfft/2 : nfft/2-1);
    [~, idx] = max(abs(cc));
    lag = lags(idx);
    tau = lag / fs;
end