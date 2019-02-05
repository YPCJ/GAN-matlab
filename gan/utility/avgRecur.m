function net = avgRecur(net, mu, istd, momentum, opts, GD)

if strcmp(GD,'G')
   batchNorm = opts.batchNorm_G;
else
   batchNorm = opts.batchNorm_D;
end

    num_cell = length(mu);
    for i=1:num_cell
        if batchNorm(i)
            net(i).mu =momentum*net(i).mu + (1-momentum)*cell2mat(mu(i));
            net(i).istd =momentum*net(i).istd + (1-momentum)*cell2mat(istd(i));
        end
    end
end
