# DS340_Project: GAN-Generated Fashion

## Overview
This project explores the use of **Generative Adversarial Networks (GANs)** to create synthetic fashion images based on the **Fashion-MNIST** dataset.  
It was developed as a final project for **DS340 (Intro to Machine Learning and AI)**.

We trained several GAN models, experimented with data augmentation techniques, and evaluated model performance using the **Fréchet Inception Distance (FID)** score.

## Project Structure
- `Augmented_GAN.ipynb`: GAN training with data augmentation.
- `Conditional_GAN.ipynb`: Conditional GAN (cGAN) model where generation is conditioned on clothing type.
- `GAN_FID.ipynb`: Calculates and analyzes FID scores to evaluate image generation quality.
- `fid_score.py`: Functions for computing FID score.
- `inception.py`: Helper file to build an InceptionV3 model for FID calculation.
- `lenet.py`: Helper file to build a LeNet model for classification tasks.
- `fashion-mnist_test.csv`: Test dataset (Fashion-MNIST).
- `models.zip`: Pretrained model weights.

## How to Run
1. Install required packages:
   ```bash
   pip install torch torchvision numpy matplotlib scipy

2. Open the notebooks (`.ipynb`) and run them sequentially:
   - Start with `Augmented_GAN.ipynb` or `Conditional_GAN.ipynb` for training.
   - Use `GAN_FID.ipynb` to calculate FID scores for evaluation.

> **Note**: Training GANs can be resource-intensive. Using **GPU acceleration** is highly recommended.

## Results
- Successfully generated realistic-looking fashion items across different categories.
- Lower FID scores were achieved with data augmentation and conditional generation.
- Demonstrated the potential of GANs to create diverse and high-quality synthetic datasets for fashion applications.

## Future Work
- Experiment with more advanced GAN architectures (e.g., **StyleGAN**, **CycleGAN**).
- Apply transfer learning on larger, real-world fashion datasets.
- Improve conditioning to allow for **multi-attribute generation** (e.g., clothing color, texture).

## Acknowledgments
- Fashion-MNIST dataset creators
- DS340 Professors and TAs
- PyTorch and torchvision open-source communities




# DS340_Project
DS340 Final Project: GAN-Generated Fashion


To calculate the FID score for both the standard GAN and the GAN trained on augmented images, run the following command in the SCC terminal:
python fid_score.py --true /projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/real_images_fid.npy --fake /projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/fake_images_fid.npy --gpu [GPU_ID]

Make sure to replace [GPU_ID] with the correct GPU number you are allocated (e.g., 0, 1, etc.). You can check which GPU you are using with the nvidia-smi command or based on your job submission details.

The .npy files for the augmented GAN are located in:
/projectnb/ds340/projects/leilani_hannah_final_project/fid_images_augmented/

The .npy files for the standard GAN are located in:
/projectnb/ds340/projects/leilani_hannah_final_project/fid_images



