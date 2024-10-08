% load dMRI data
s = rng;
dwi0 = load_untouch_nii('');
dwi0 = dwi0.img;
% simulate 3D spatially variant random noise to mimic parallel imaging with
% the characteristic of fast variations in x, and slow in y and z
% dimensions.
sm1 = customgauss([size(dwi0,1),size(dwi0,2)], round(0.5*size(dwi0,2)), round(0.5*size(dwi0,2)), 0, 0.2, 1, [1 1]);
sm1 = repmat(sm1,[1 1 size(dwi0,3)]);
sm1_z = customgauss([size(dwi0,1),size(dwi0,3)], round(0.7*size(dwi0,1)), round(0.7*size(dwi0,1)), 0, 0.1, 1, [1 1]);
sm1_z = sm1_z(floor(size(sm1_z,1)/2),:);
sm1 = sm1.*repmat(reshape(sm1_z,[1 1 size(dwi0,3)]),[size(dwi0,1),size(dwi0,2)]);
% add additional sinusoidal variation along x dimension
sm2 = sin(repmat(linspace(-5*pi,5*pi,size(dwi0,2)), size(dwi0,1),1));
sm = sm1 + 0.1*repmat(sm2,[1 1 size(dwi0,3)]);
imshow(sm(:,:,1))
% normalize the std to 1
sm = sm./max(sm(:));
sm_ = permute(sm,[2,1,3]);
% non-stationary noise level map
sigma_map = 0.03*sm_;
sigma_map_nii = make_nii(sigma_map);
save_nii(sigma_map_nii,'')