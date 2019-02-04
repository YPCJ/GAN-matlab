try
   gpuDevice;
   gcount = 1;
catch err
   disp('no gpu available, use cpu instead');
   gcount = 0;
end
if gcount > 0
   gpuobj = gpuDevice(num_gpu);
   gpuDevice(gpuobj.Index);
   opts.isGPU = 1;
   opts.eval_on_gpu = 1;
else
   opts.isGPU = 0;
   opts.eval_on_gpu = 0;
   disp('GPU is not available, using CPU.')
end

clear err gcount