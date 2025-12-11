%% MVDR (3D steering: azimuth + elevation) using RIR delays only (no manual delay)
clear; close all; clc;

%% ========== 1. Read mic positions (assume mic_positions.xlsx has X Y columns) ==========
data = readmatrix('mic_positions.xlsx');
X = data(:,1); Y = data(:,2);
Z = zeros(size(X));           % 如果你在Excel里有Z列，把这行改为 Z = data(:,3);
mic_pos = [X Y Z] / 1000;     % 假定Excel单位为 mm，直接换成 m；如果是 cm 用 /100

% center to given array center (optional)
array_center = [2.5 2 1.5];   % 房间位置（m）
mic_pos = mic_pos - mean(mic_pos,1) + array_center;

Nmic = size(mic_pos,1);
fprintf('Nmic = %d, mic positions loaded (meters).\n', Nmic);

%% ========== 2. Room, RIR and signals parameters ==========
c = 340;
fs = 16000;
t = (0:1/fs:1-1/fs)';
f0 = 1000;         % target freq (Hz)
f1 = 2000;         % interference freq (Hz)
SNR = 10;          % dB
INR = 10;          % dB (interference relative to target amplitude)
nsample = 4096;    % RIR length

% source 3D positions (if you prefer follow-up: can set distance+az+el)
s_target = [2 3.5 2];   % if you set these, they'll be used for RIR generation (m)
s_interf  = [2.5 3.8 2];

% Alternatively specify azimuth/elevation + distance:
% az, el in degrees (az=0 along +X, increasing toward +Y)
az_target = 0; el_target = 0; dist_target = norm(s_target - mean(mic_pos,1));
az_interf  = 60; el_interf  = 0; dist_interf  = norm(s_interf - mean(mic_pos,1));

%% ========== 3. Fraunhofer check (far-field criterion) ==========
D = 0.15;           % 阵列最大孔径 (m) — 150 mm
f_max = 5000;       % 你关心的最高频率（Hz）, 本例 5 kHz
lambda_min = c / f_max;
R_far = 2 * D^2 / lambda_min;   % Fraunhofer distance (approx)

fprintf('Array diameter D=%.3f m, f_max=%d Hz, lambda_min=%.3f m\n', D, f_max, lambda_min);
fprintf('Fraunhofer distance R_far ≈ %.3f m\n', R_far);

% compute actual distances of sources to array center
r_center = mean(mic_pos,1);
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
beta = 0.4;     % wall reflection coefficient
mtype = 'omnidirectional';  % 宏观选择，取决于你的rir_generator是否支持
order = -1;
dim = 3;
orientation = [pi/2 0];
hp_filter = 1;

h_target = zeros(Nmic, nsample);
h_interf = zeros(Nmic, nsample);

fprintf('Generating RIRs for %d microphones (this may take a while)...\n', Nmic);
for m = 1:Nmic
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

Nt = length(x_target);
X_target = zeros(Nt, Nmic);
X_interf  = zeros(Nt, Nmic);

% convolution (same length)
for m = 1:Nmic
    X_target(:,m) = conv(x_target, h_target(m,:), 'same');
    X_interf(:,m)  = conv(x_interf,  h_interf(m,:),  'same');
end

% scale interference by INR (dB)
X_interf = X_interf / (10^(INR/20));

% add noise to meet overall SNR per channel (relative to target)
sig_rms = sqrt(mean(X_target.^2, 1));
desired_noise_rms = sig_rms ./ (10^(SNR/20));
noise = randn(size(X_target));
cur_noise_rms = sqrt(mean(noise.^2, 1));
noise = noise .* (desired_noise_rms ./ cur_noise_rms);

X_noisy = X_target + X_interf + noise;
fprintf('Signals prepared (RIR only). No extra delays added.\n');

%% ========== 6. STFT parameters and compute STFT per channel ==========
Nfft = 1024;
win = hamming(512);
noverlap = 256;

% compute STFT channel-wise; we store as S(freq, time, channel)
for m = 1:Nmic
    [S(:,:,m), F, T] = stft(X_noisy(:,m), fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
end
Fbins = length(F);
Tframes = length(T);

% limit up to f_limit (0-5kHz)
f_limit = find(F <= 5000, 1, 'last');
fprintf('STFT computed: %d freq bins, %d time frames. Using bins 1..%d up to %.1f Hz.\n', Fbins, Tframes, f_limit, F(f_limit));

%% ========== 7. Build steering vectors (per-frequency) - 3D handling ==========
a_target = zeros(Nmic, Fbins);   % complex steering (phase / amplitude)
a_interf  = zeros(Nmic, Fbins);

% compute unit direction vector from az,el when using far-field plane-wave
deg2rad = @(x) x*pi/180;
d_target = [cos(deg2rad(el_target))*cos(deg2rad(az_target));
            cos(deg2rad(el_target))*sin(deg2rad(az_target));
            sin(deg2rad(el_target))];   % 3x1
d_interf = [cos(deg2rad(el_interf))*cos(deg2rad(az_interf));
            cos(deg2rad(el_interf))*sin(deg2rad(az_interf));
            sin(deg2rad(el_interf))];

for k = 1:Fbins
    freq = F(k);
    if k > f_limit
        a_target(:,k) = zeros(Nmic,1);
        a_interf(:,k)  = zeros(Nmic,1);
        continue;
    end

    if use_farfield_target
        % plane-wave (phase only) using dot(mic_pos, d)
        % note: use mic_pos * d_target -> Nmic x 1 vector of projections
        tau_proj = mic_pos * d_target;   % Nmic x 1
        a_target(:,k) = exp(-1j*2*pi*freq*tau_proj / c);   % phase-only
    else
        % near-field: spherical wave: include distance-based phase and amplitude 1/r
        for m = 1:Nmic
            dist = norm(mic_pos(m,:) - s_target);   % distance mic->source
            a_target(m,k) = exp(-1j*2*pi*freq*dist/c) / max(dist, 1e-6);
        end
    end

    if use_farfield_interf
        tau_proj_i = mic_pos * d_interf;
        a_interf(:,k) = exp(-1j*2*pi*freq*tau_proj_i / c);
    else
        for m = 1:Nmic
            disti = norm(mic_pos(m,:) - s_interf);
            a_interf(m,k) = exp(-1j*2*pi*freq*disti/c) / max(disti, 1e-6);
        end
    end
end

%% ========== 8. Memory-safe per-frame MVDR (diagonal loading adapted) ==========
fprintf('Running MVDR (per-frequency, per-frame) ...\n');
Yf = zeros(Fbins, Tframes);   % complex freq x time result
% Mavg = 5;                     % averaging window for covariance
% epsilon = 1e-6;               % base loading

Mavg = 11;                     % averaging window for covariance
epsilon = 1e-3;               % base loading

tic
for k = 1:f_limit
    Xkf = squeeze(S(k,:,:)).';   % Nmic x Tframes   (check stft order; this matches earlier)
    at = a_target(:,k);
    if norm(at)==0
        continue;
    end

    for tt = 1:Tframes
        t1 = max(1, tt - floor(Mavg/2));
        t2 = min(Tframes, tt + floor(Mavg/2));
        Xloc = Xkf(:, t1:t2);
        Rxx = (Xloc * Xloc') / size(Xloc,2);

        % adaptive diagonal loading proportional to trace
        reg = epsilon * trace(Rxx) / Nmic;
        Rxx = Rxx + reg * eye(Nmic);

        % MVDR weight: w = R^{-1} a / (a^H R^{-1} a)
        w = Rxx \ at;
        denom = (at' * w);
        if abs(denom) < 1e-12
            W = zeros(size(w));
        else
            W = w ./ denom;
        end

        % output for frame tt
        Yf(k, tt) = W' * Xkf(:, tt);
    end
end
toc

%% ========== 9. ISTFT and normalization ==========
y_mvdr = istft(Yf, fs, 'Window', win, 'OverlapLength', noverlap, 'FFTLength', Nfft);
% istft may return a slightly different length; trim/pad to original length
y_mvdr = real(y_mvdr);

% robust length alignment
if length(y_mvdr) >= Nt
    y_mvdr = y_mvdr(1:Nt);
else
    y_mvdr = [y_mvdr; zeros(Nt - length(y_mvdr), 1)];
end


% energy normalization (avoid clipping)
y_mvdr = y_mvdr / max(abs(y_mvdr) + 1e-12);

%% ========== 10. Diagnostics: show waveforms & spectrograms & PSD around 1kHz ==========
figure(1); clf;
subplot(3,1,1); plot((0:Nt-1)/fs, X_target(:,1)); title('Target (mic 1, with RIR)'); xlabel('Time (s)');
subplot(3,1,2); plot((0:Nt-1)/fs, X_noisy(:,1));  title('Noisy (mic 1)'); xlabel('Time (s)');
subplot(3,1,3); plot((0:Nt-1)/fs, y_mvdr);         title('MVDR output (time)'); xlabel('Time (s)');

figure(2); clf;
subplot(3,1,1); spectrogram(X_target(:,1), hamming(256), 128, 512, fs, 'yaxis'); title('Target spectrogram (mic1)');
subplot(3,1,2); spectrogram(X_noisy(:,1), hamming(256), 128, 512, fs, 'yaxis'); title('Noisy spectrogram (mic1)');
subplot(3,1,3); spectrogram(y_mvdr,         hamming(256), 128, 512, fs, 'yaxis'); title('MVDR output spectrogram');

% PSD around 1 kHz
nfft_psd = 4096;
[Pxx_in, Fp]  = pwelch(X_noisy(:,1), hann(1024), 512, nfft_psd, fs);
[Pxx_out, ~]  = pwelch(y_mvdr,        hann(1024), 512, nfft_psd, fs);

figure(3); clf;
plot(Fp, 10*log10(Pxx_in + eps)); hold on;
plot(Fp, 10*log10(Pxx_out + eps), 'LineWidth', 1.5);
xlim([800 1200]);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
legend({'Mic1 noisy','MVDR output'}); title('PSD around 1 kHz');

%% ========== 11. Save output audio ==========
audiowrite('y_mvdr.wav', y_mvdr, fs);
fprintf('Saved MVDR output to y_mvdr.wav\n');

%% ========== End ==========
fprintf('Processing complete. Check spectrogram and PSD plot: 1 kHz should be visible/enhanced.\n');
