function finish_flag = CoreLDPCDecoderFile(args)

% Flag
finish_flag = 0;

% Unpack arguments
filename_in  = args.filename_in;
filename_out = args.filename_out;

% Load file
contents = load(filename_in);
% Extract variables
llr_input = contents.llr_input;
code_type = contents.code_type;

% Input needs to be 3D
assert(numel(size(llr_input)) == 3);

% Organized as (SNR, num_packets, N)
num_snr_points = size(llr_input, 1);
num_codewords  = size(llr_input, 2);
N_input        = size(llr_input, 3);

% Interleaver seed - do NOT generally change
inter_seed = 1111;

% Code parameters
if strcmp(code_type, 'ldpc')
    % Taken from IEEE 802.11n: HT LDPC matrix definitions
    % You can change this according to your needs
    Z = 27;
    rotmatrix = ...
        [0 -1 -1 -1 0 0 -1 -1 0 -1 -1 0 1 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        22 0 -1 -1 17 -1 0 0 12 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1 -1;
        6 -1 0 -1 10 -1 -1 -1 24 -1 0 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1 -1;
        2 -1 -1 0 20 -1 -1 -1 25 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1 -1;
        23 -1 -1 -1 3 -1 -1 -1 0 -1 9 11 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1 -1;
        24 -1 23 1 17 -1 3 -1 10 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1 -1;
        25 -1 -1 -1 8 -1 -1 -1 7 18 -1 -1 0 -1 -1 -1 -1 -1 0 0 -1 -1 -1 -1;
        13 24 -1 -1 0 -1 8 -1 6 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1 -1;
        7 20 -1 16 22 10 -1 -1 23 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1 -1;
        11 -1 -1 -1 19 -1 -1 -1 13 -1 3 17 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0 -1;
        25 -1 8 -1 23 18 -1 14 9 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 0;
        3 -1 -1 -1 16 -1 -1 2 25 5 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0];

    H = zeros(size(rotmatrix)*Z);
    Zh = diag(ones(1,Z),0);

    % Convert into binary matrix
    for r=1:size(rotmatrix,1)
        for c=1:size(rotmatrix,2)
            rotidx = rotmatrix(r,c);
            if (rotidx > -1)
                Zt = circshift(Zh,[0 rotidx]);
            else
                Zt = zeros(Z);
            end
            limR = (r-1)*Z+1:r*Z;
            limC = (c-1)*Z+1:c*Z;
            H(limR,limC) = Zt;
        end
    end

    hDec = comm.LDPCDecoder('ParityCheckMatrix', sparse(H), 'DecisionMethod', 'Soft decision', ...
        'IterationTerminationCondition', 'Parity check satisfied', 'MaximumIterationCount', 50);

    % System parameters
    K = size(H, 1);
    N = size(H, 2);
elseif strcmp(code_type, 'polar')
    % Polar code parameters
    K = 128;
    N = 256;
    L = 4; % List length
else
    error('Invalid code type!')
end

% Depending on code type, assert that the size is right
padded_llrs = N - N_input;
assert(padded_llrs >= 0);

% Interleaver
rng(inter_seed);
P = randperm(N);
R(P) = 1:N;

% Output
bits_out = zeros(num_snr_points, num_codewords, K);

% Progress
% progressbar(0, 0);

% For each SNR point
for snr_idx = 1:num_snr_points
    % For each codeword
    for codeword_idx = 1:num_codewords
        % Fetch LLRs
        local_llr = squeeze(llr_input(snr_idx, codeword_idx, :));
        
        % If needed, pad with erasures (complete unknowns)
        local_llr = [local_llr; zeros(padded_llrs, 1)]; %#ok<AGROW>
        
        % Deinterleave bits
        llrDeint = double(local_llr(R));
        
        % Channel decoder
        if strcmp(code_type, 'ldpc')
            llrOut = hDec(llrDeint);
            bitsEst = (sign(-llrOut) +1) / 2;
        elseif strcmp(code_type, 'polar')
            bitsEst = nrPolarDecode(llrDeint, K, N, L);
        end
        
        % Store
        bits_out(snr_idx, codeword_idx, :) = bitsEst;
        % Progress
%         progressbar(codeword_idx / num_codewords, []);
    end
    % Progress
%     progressbar([], snr_idx / num_snr_points);
end

% Write bits to output file
save(filename_out, 'bits_out');

% progressbar(1);

% Flag
finish_flag = 1;

end