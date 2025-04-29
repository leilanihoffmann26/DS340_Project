# DS340_Project
DS340 Final Project: GAN-Generated Fashion


To calculate the FID score for both the standard GAN and the GAN trained on augmented images, run the following command in the SCC terminal:
python fid_score.py --true /projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/real_images_fid.npy --fake /projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/fake_images_fid.npy --gpu [GPU_ID]

Make sure to replace [GPU_ID] with the correct GPU number you are allocated (e.g., 0, 1, etc.). You can check which GPU you are using with the nvidia-smi command or based on your job submission details.

The .npy files for the augmented GAN are located in:
/projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/

The .npy files for the standard GAN are located in:
/projectnb/ds340/projects/leilani_hannah_final_project/fid_images



