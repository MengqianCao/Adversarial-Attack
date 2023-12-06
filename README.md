# Overview
The objective of this project is to investigate the adversarial attack by adding perturbation to the target objects and finally disrupt the model’s recognition to these objects. There are two types of adversarial attacks this project mainly focuses on: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD).</br></br>
The project involves implementing both FGSM and PGD attacks, basic factors such as the effectiveness, perturbation magnitude, and robustness of two attacks will be considered to make a comparison between these two attacks. Additionally, this project will implement a defense mechanism to the trained model through adversarial training to mitigate the impact of adversarial attacks, and visualizations and results will be provided in the end.
# Dataset & Model
The project uses Cifar10 dataset as the test benchmark, and ResNet-50 is chosen to be the base model for classifying test images. 

# Project Structure

## Root Directory
- `adv_train.py`: Code for adversarial training.
- `best_fgsm_model.pth`: Checkpoint for the best model trained with FGSM.
- `best_model.pth`: Checkpoint for the best model.
- `best_pgd_model.pth`: Checkpoint for the best model trained with PGD.
- `README.md`: Instructions on how to run the program.
- `requirements.txt`: List of required libraries.
- `test_attack.py`: Code to assess the success rate of attacks.
- `train.py`: Code for training the ResNet image classifier.

## Data Directory
- `data/`
  - `cifar10/`
    - `cifar-10-batches-py/`
      - `batches.meta`
      - `data_batch_1`
      - `data_batch_2`
      - `data_batch_3`
      - `data_batch_4`
      - `data_batch_5`
      - `readme.html`
      - `test_batch`

## Tools Directory
- `tools/`
  - `attack.py`: Code for FGSM and PGD adversarial attack methods.
  - `conduct.py`: Code for inference, including training, testing, and adversarial training.
  - `dataloader.py`: Code using PyTorch's DataLoader for loading and preprocessing the CIFAR-10 dataset.
  - `visualize.py`: Code for various visualization methods, including the `plot_adv_images()` function for testing adversarial attacks.
# Usage
## To train the classificator
```shell
python train.py
```
Run the above command in the terminal, and you can view the logs during training. After the training is completed, you can find the checkpoint file in the default path and obtain the curve graphs of train loss and accuracy.
## To perform adversarial training
```shell
python adv_train.py [attack_function]
```
Where [attack_function] means the method used to generate adversarial samples, whose value is fgsm or pgd. </br></br>
For example, if you want to use FGSM method to generate the adversarial samples in training, please run the following command in the terminal.
```shell
python adv_train.py fgsm
```
## To test attack results
```shell
python test_attack.py [attack_function] [model_path]
```
Where [attack_function] means the method used to execute attack, whose value is fgsm or pgd. [model_path] means the model used for testing.</br></br>
For example, if you want to use the fgsm method to attack the model which trained with fgsm adversarial training, please run the following command in the terminal.
```shell
python test_attack.py fgsm best_fgsm_model.pth
```
Run the above command in the terminal, and you can view the logs during testing with different epsilons. After the testing is completed, you can obtain the curve graph of attack success rate with different epsilons, and a plot of visualized examples.