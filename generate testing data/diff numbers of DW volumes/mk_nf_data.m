addpath('NIfTI_20140122')
% load bvec 
% the number of DW volumes can be set to 9, 12, 15, 22 and 32
bvec = load('');
if size(bvec,2)==3
    bvec = bvec';
end
% load b0 and diffusion tensor
data = load_untouch_nii('');
data = data.img;
b0 = data(:,:,:,1);
b0 = (b0 - min(b0(:)))./(max(b0(:))-min(b0(:)));
tensor = data(:,:,:,2:7);
% generate noise-free data
dwi = zeros(size(data,1),size(data,2),size(data,3),size(bvec,2));
for i = 1:size(dwi,1)
    for j = 1:size(dwi,2)
        for k = 1:size(dwi,3)
            for p = 1:size(dwi,4)
                D = [tensor(i,j,k,1),tensor(i,j,k,4),tensor(i,j,k,5);tensor(i,j,k,4),tensor(i,j,k,2),tensor(i,j,k,6);tensor(i,j,k,5),tensor(i,j,k,6),tensor(i,j,k,3)];
                dwi(i,j,k,p) = b0(i,j,k)*exp(-1000*(bvec(:,p)')*D*bvec(:,p));
            end
        end
    end
end
dwi_nii = make_nii(dwi);
save_nii(dwi_nii,'')