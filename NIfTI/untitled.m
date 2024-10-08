data_fileFolder=fullfile('/home/chenyunwei/sph_unet_test/test_1/data/gt');
data_dirOutput=dir(fullfile(data_fileFolder,'*'));
data_fileNames={data_dirOutput.name};
bvec = load('/home/chenyunwei/sph_unet_test/test_2/data/bvec/bvec_11.txt');
g = bvec(2:size(bvec,1),:)';
for i = 3:16
    data_fileNames{i}
    path = strcat('/home/chenyunwei/sph_unet_test/test_1/data/gt/',data_fileNames{i});
    tensor_path = strcat(path,'/tensor_cor.nii.gz');
    dwi_b0_path = strcat(path,'/recon.nii.gz');
    tensor = load_untouch_nii(tensor_path);
    tensor = tensor.img;
    dwi_b0 = load_untouch_nii(dwi_b0_path);
    dwi_b0 = dwi_b0.img;
    b0 = dwi_b0(7:134,7:134,:,1);
    b1000 = zeros([128,128,96,size(g,2)]);
    for i1 = 1:128
        for j1 = 1:128
            for k1 = 1:96
                for p1 = 1:size(g,2)
                    D = [tensor(i1,j1,k1,1),tensor(i1,j1,k1,4),tensor(i1,j1,k1,6);tensor(i1,j1,k1,4),tensor(i1,j1,k1,2),tensor(i1,j1,k1,5);tensor(i1,j1,k1,6),tensor(i1,j1,k1,5),tensor(i1,j1,k1,3)];
                    b1000(i1,j1,k1,p1) = b0(i1,j1,k1)*exp(-1000*(g(:,p1)')*D*g(:,p1));
                end
            end
        end
    end
    DWI = zeros(128,128,96,size(bvec,1));
    DWI(:,:,:,1) = b0;
    DWI(:,:,:,2:size(bvec,1)) = b1000;
    DWI_ = (DWI - min(DWI(:)))/(max(DWI(:))-min(DWI(:)));
    DWI_nii = make_nii(DWI_);
    save_path = strcat('/home/chenyunwei/sph_unet_test/test_2/data/gt/',data_fileNames{i},'/dwi_11.nii.gz');
    save_nii(DWI_nii,save_path)
end