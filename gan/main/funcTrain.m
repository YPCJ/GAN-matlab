function [model_D, model_G, perf_D, perf_G] = funcTrain(noise, real, opts)
net_struct_G = opts.net_struct_G;
net_struct_D = opts.net_struct_D;

isGPU = opts.isGPU;
num_samples = size(noise,1);

fprintf('\nNum of Training Samples:%d\n',num_samples);
disp(['Generator struct : ' num2str(net_struct_G)]);
disp(['Discriminator struct : ' num2str(net_struct_D)]);
disp(size(real));

perf_D = zeros(opts.max_epoch,1);
perf_G = zeros(opts.max_epoch,1);
start_epoch = opts.epoch;

switch opts.isPretrain
    case 0
        disp('Weight initialization.')
        isSparse = 0; isNorm = 1;
        pre_net_D = randInitNet(net_struct_D, isSparse, isNorm, isGPU, opts.initial, opts.batchNormlization);
        pre_net_G = randInitNet(net_struct_G, isSparse, isNorm, isGPU, opts.initial, opts.batchNormlization);
    case 2
        load([opts.save_model_path 'resume_param.mat']);
        disp([opts.save_model_path 'resume_param.mat loaded.'])
        load([opts.save_model_path num2str(epoch) '.mat']);
        disp([opts.save_model_path num2str(epoch) '.mat loaded.'])
        pre_net_G = model_G;
        pre_net_D = model_D;
        start_epoch = epoch+1;
end

disp(['isGPU : ' num2str(isGPU) ' / ' opts.cost_function ' / ' opts.learner '(' num2str(opts.learner_scale) ') / Batch Normalization : ' num2str(opts.batchNormlization)])

net_iterative_D = pre_net_D;
net_iterative_G = pre_net_G;

batch_id = genBatchID(num_samples,opts.batch_size);
num_batch = size(batch_id,2);

if start_epoch == 1
    net_weights_inc_D = zeroInitNet(net_struct_D, opts.isGPU);
    net_weights_inc_G = zeroInitNet(net_struct_G, opts.isGPU);
    
    net_grad_ssqr_D = zeroInitNet(net_struct_D, opts.isGPU, eps);
    net_grad_ssqr_G = zeroInitNet(net_struct_G, opts.isGPU, eps);
    
    net_grad_ssqr_D = struct('W',{net_grad_ssqr_D.W},'b', {net_grad_ssqr_D.b});
    net_grad_ssqr_G = struct('W',{net_grad_ssqr_G.W},'b', {net_grad_ssqr_G.b});
    
    if strcmp(opts.learner, 'adam')
        [net_grad_ssqr_D(:).W2] = deal(net_grad_ssqr_D.W); [net_grad_ssqr_D(:).b2] = deal(net_grad_ssqr_D.b);
        [net_grad_ssqr_G(:).W2] = deal(net_grad_ssqr_G.W); [net_grad_ssqr_G(:).b2] = deal(net_grad_ssqr_G.b);
    end
    if opts.batchNormlization == 1
        [net_grad_ssqr_D(:).gamma] = deal(net_grad_ssqr_D.b); [net_grad_ssqr_D(:).beta]  = deal(net_grad_ssqr_D.b);
        [net_grad_ssqr_G(:).gamma] = deal(net_grad_ssqr_G.b); [net_grad_ssqr_G(:).beta]  = deal(net_grad_ssqr_G.b);
        [net_weights_inc_D(:).gamma] = deal(net_weights_inc_D.b); [net_weights_inc_D(:).beta]  = deal(net_weights_inc_D.b);
        [net_weights_inc_G(:).gamma] = deal(net_weights_inc_G.b); [net_weights_inc_G(:).beta]  = deal(net_weights_inc_G.b);
        if strcmp(opts.learner, 'adam')
            [net_grad_ssqr_D(:).gamma2] = deal(net_grad_ssqr_D.b); [net_grad_ssqr_D(:).beta2]  = deal(net_grad_ssqr_D.b);
            [net_grad_ssqr_G(:).gamma2] = deal(net_grad_ssqr_G.b); [net_grad_ssqr_G(:).beta2]  = deal(net_grad_ssqr_G.b);
        end
    end
end

for epoch = start_epoch:opts.max_epoch
    tic
    seq = randperm(num_samples);
    cost_sum_G = 0; cost_sum_D = 0;
    for bid = 1:num_batch
        perm_idx = seq(batch_id(1,bid):batch_id(2,bid));
        
        if isGPU
            batch_data = gpuArray(randn(size(perm_idx,2)*2,size(noise,2)));
            batch_label = gpuArray(real(perm_idx,:));
        else
            batch_data = randn(size(perm_idx,2)*2,size(noise,2));
            batch_label = real(perm_idx,:);
        end
        
        if epoch>opts.change_momentum_point
            momentum=opts.final_momentum;
        else
            momentum=opts.initial_momentum;
        end
        
        [cost_D, net_grad_D, ~, ~, D_real] = computeNetGradient(net_iterative_G, net_iterative_D, batch_data(1:end/2,:), batch_label, opts, 'D');
        net_grad_D_sum = net_grad_D.real;
        for ll = 1:length(net_grad_D.real)
            net_grad_D_sum(ll).W = net_grad_D.real(ll).W + net_grad_D.fake(ll).W;
            net_grad_D_sum(ll).b = net_grad_D.real(ll).b + net_grad_D.fake(ll).b;
            
            if opts.batchNormlization && opts.batchNorm_D(ll) == 1
                net_grad_D_sum(ll).gamma = net_grad_D.real(ll).gamma + net_grad_D.fake(ll).gamma;
                net_grad_D_sum(ll).beta  = net_grad_D.real(ll).beta  + net_grad_D.fake(ll).beta;
            end
        end
        [net_iterative_D, net_weights_inc_D, net_grad_ssqr_D] = learner(net_iterative_D, momentum, net_weights_inc_D, net_grad_ssqr_D, net_grad_D_sum, opts, epoch, bid, num_batch, 'D');
        
        [cost_G,net_grad_G, G_fake, D_fake, ~] = computeNetGradient(net_iterative_G, net_iterative_D, batch_data(end/2+1:end,:), batch_label, opts, 'G');
        [net_iterative_G, net_weights_inc_G, net_grad_ssqr_G] = learner(net_iterative_G, momentum, net_weights_inc_G, net_grad_ssqr_G,     net_grad_G, opts, epoch, bid, num_batch, 'G');
        
        if rem(bid, opts.checker) == 0
            figure(1), clf('reset'); 

            subplot(4,2,1), imshow((reshape(G_fake(1,:),[28 28])*1)), title(num2str(D_fake(1)));
            subplot(4,2,3), imshow((reshape(G_fake(2,:),[28 28])*1)), title(num2str(D_fake(2)));
            subplot(4,2,5), imshow((reshape(G_fake(3,:),[28 28])*1)), title(num2str(D_fake(3)));
            subplot(4,2,7), imshow((reshape(G_fake(4,:),[28 28])*1)), title(num2str(D_fake(4)));
            
            subplot(4,2,2), imshow((reshape(batch_label(1,:),[28 28])*1)), title(num2str(D_real(1)));
            subplot(4,2,4), imshow((reshape(batch_label(2,:),[28 28])*1)), title(num2str(D_real(2)));
            subplot(4,2,6), imshow((reshape(batch_label(3,:),[28 28])*1)), title(num2str(D_real(3)));
            subplot(4,2,8), imshow((reshape(batch_label(4,:),[28 28])*1)), title(num2str(D_real(4)));

            drawnow;
        end
        
        cost_sum_G = cost_sum_G + cost_G;
        cost_sum_D = cost_sum_D + cost_D;
    end
    perf_D(epoch) = gather(cost_sum_D);
    perf_G(epoch) = gather(cost_sum_G);
    toc
    
    fprintf('D cost : %.3f / G cost : %.3f\n', cost_sum_D, cost_sum_G)
    model_G = net_iterative_G;
    model_D = net_iterative_D;
    save([opts.save_model_path num2str(epoch) '.mat'], 'model_G','model_D', 'opts');
    disp([opts.save_model_path num2str(epoch) '.mat saved.'])
    save([opts.save_model_path 'resume_param.mat'],'net_grad_ssqr_D','net_grad_ssqr_G','net_weights_inc_D','net_weights_inc_G','epoch','perf_D','perf_G');
end