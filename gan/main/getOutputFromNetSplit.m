function output = getOutputFromNetSplit(net,data,opts)

num_samples = size(data,1);
dim_output = opts.net_struct(end);
chunck_size = ceil(num_samples/opts.eval_on_gpu_num_split);

output = zeros(num_samples,dim_output,'single');
disp(['opts.eval_on_gpu : ' num2str(opts.eval_on_gpu)])

% if opts.eval_on_gpu
%     output = gpuArray.zeros(num_samples,dim_output,'single');
%     disp(['opts.eval_on_gpu : ' num2str(opts.eval_on_gpu)])
% else
%     output = zeros(num_samples,dim_output,'single');
% end

for i = 1:opts.eval_on_gpu_num_split
    idx_start = (i-1)*chunck_size+1;
    idx_end = i*chunck_size;
    if idx_end>num_samples; idx_end = num_samples; end
    output_chunck = getOutputFromNet(net,data(idx_start:idx_end,:),opts);
    if opts.eval_on_gpu && opts.isGPU %existsOnGPU(output_chunck)
        output(idx_start:idx_end,:) = gather(output_chunck);
    else
        output(idx_start:idx_end,:) = output_chunck;
    end
end