function mse = getMSE(gnd, predlb, opts, cost)
    switch cost
        case 'mse'
            mse = mean( mean( (gnd-predlb).^2 ) );
        case 'mse2'
            mse = mean( mean( (gnd(:,1:end/2)-predlb(:,1:end/2)).^2 ) + mean( (gnd(:,end/2+1:end)-predlb(:,end/2+1:end)).^2 ));
        case 'beta1'
            tmpa = repmat(opts.beta1_param_a_1(1,:), [size(gnd,1), 1]);
            tmpb = repmat(opts.beta1_param_b_1(1,:), [size(gnd,1), 1]);

            gnd(gnd<opts.beta1_extremes1) = opts.beta1_extremes1;
            gnd(gnd>opts.beta1_extremes2) = opts.beta1_extremes2;
            tmp = gnd-predlb;

            mse = mean( mean( tmp.^2 - repmat(var(tmp),[size(tmp,1) 1]) .* (tmpa.*log(gnd) + tmpb.*log(1-gnd))));
        case 'beta2'
            tmpa = repmat(opts.beta1_param_a_1(1,:), [size(gnd,1), 1]);
            tmpb = repmat(opts.beta1_param_b_1(1,:), [size(gnd,1), 1]);

            gnd(gnd<opts.beta1_extremes1) = opts.beta1_extremes1;
            gnd(gnd>opts.beta1_extremes2) = opts.beta1_extremes2;
            tmp = (gnd-predlb).^2;

            mse = mean( mean( tmp - sum(sum(tmp))/numel(gnd) * (tmpa.*log(gnd) + tmpb.*log(1-gnd))));
    end
end