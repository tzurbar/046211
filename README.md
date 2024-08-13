# Skin Diseases Calssification with pretrained models
The aim of the project is to develop a system that allows individuals to get a preliminary analysis of skin problems using an image, serving as a pre-consultation with a medical professional.

## Method <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/entrepreneurship_8552566.png" alt="Icon" width="30" height="30">
In this project we examined the use of pretrained models – DINOV2 (ViT) and ResNet-101 (CNN). We aimed to see if we can get better results using large pretrained models with fine tuning methods (DoRA).

<div align="center">
    <img src="https://github.com/tzurbar/046211/blob/main/Assets/Block_diagram.png?raw=true" alt="Block Diagram">
    <p><strong>Figure 1:</strong> Block Diagram.</p>
</div>

## Dataset <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/book-stack_3389081.png" alt="Icon" width="30" height="30">
We used a [Kaggle dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data) that contains 27,153 skin disease labeled images with 10 different disease classes:
1.	Eczema – 1,677 images
2.	Melanoma – 15,750 images
3.	Atopic Dermatitis – 1,250 images
4.	Basal Cell Carcinoma (BCC) – 3,323 images
5.	Melanocytic Nevi (NV) – 7,970 images
6.	Benign Keratosis-like Lesions (BKL) – 2,624 images
7.	Psoriasis pictures Lichen Planus and related diseases – 2,000 images
8.	Seborrheic Keratoses and other Benign Tumors – 1,800 images
9.	Tinea Ringworm Candidiasis and other Fungal Infections – 1,700 images
10.	Warts Molluscum and other Viral Infections – 2,103 images

<div align="center">
    <img src="https://github.com/tzurbar/046211/blob/main/Assets/Classes_images.png?raw=true" alt="Block Diagram">
    <p><strong>Figure 2:</strong> Classes images.</p>
</div>

## Prerequisites <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/brain_13378620.png" alt="Icon" width="30" height="30">
| Library       | Version |
| ------------- | ------- |
| Torch         | 2.3.1   |
| Torchvision   | 0.18.1  |
| NumPy         | 1.26.4  |
| Pandas        | 2.1.4   |
| Matplotlib    | 3.7.1   |
| Scikit-learn  | 1.3.2   |
| Seaborn       | 0.13.1  |
| kornia        | 0.7.3   |

## Files in the repository <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/document_11456210.png" alt="Icon" width="30" height="30">
| File name	    | Purpsoe |
| ------------- | ------- |
| DINOv2_with_DORA_with_augmentation.ipynb         | Traning, validating and testing of the DINOV2 with DoRA   |
| ResNet_with_DORA_with_augmentation.ipynb        | Traning, validating and testing of the ResNet-101 with DoRA   |
| Plot_image_classes.ipynb   | Plot Classes Images from our dataset  |

## How to use <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/book_16658438.png" alt="Icon" width="30" height="30">
### Setup Instructions
To run the code, follow these steps:

- **Download the Dataset**:
  - Go to the [Kaggle dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data) and download the dataset. You can use Kaggle's API or download it manually.

- **Upload to Google Drive**:
  - Upload the downloaded classes folders to your Google Drive account.
  - Place the classes folders in the directory `046211/project` within your Google Drive.

  - Ensure that the path to the dataset in your code matches the Google Drive directory:
    ```python
    dataset_path = '/content/drive/MyDrive/046211/project'
    ```
- **Run the Code**:
  - Once the classes folders is correctly placed in the specified directory and your notebook is set up to access Google Drive, you can run the code as usual.

We recommend using a GPU on Google Colab for running the code, as it is very compute-intensive.

## Results and Comparison <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/stats_7074922.png" alt="Icon" width="30" height="30">
**Comparing the two pretrained models with Dora fine-tuning results**:
| **Condition**           | **Number of Images** | **ResNet101 Precision** | **DINOV2 Precision** |
|-------------------------|----------------------|-------------------------|----------------------|
| Eczema                  | 1,677                | 0.61                    | 0.54                 |
| Melanoma                | 15,750               | 0.94                    | 0.97                 |
| Atopic Dermatitis       | 1,250                | 0.51                    | 0.46                 |
| BCC                     | 3,323                | 0.80                    | 0.83                 |
| NV                      | 7,970                | 0.86                    | 0.89                 |
| BKL                     | 2,624                | 0.67                    | 0.76                 |
| Psoriasis               | 2,000                | 0.47                    | 0.51                 |
| Seborrheic              | 1,800                | 0.61                    | 0.69                 |
| Tinea                   | 1,700                | 0.57                    | 0.69                 |
| Warts Molluscum         | 2,103                | 0.71                    | 0.69                 |
| **Weighted Accuracy**   | 27,153               | 0.74                    | 0.77                 |

We observed improved results with the DINOV2 model overall; however, the ResNet-101 model performs better in three specific classes. As expected, the classes with more images show better performance.

<div align="center">
    <img src="https://github.com/tzurbar/046211/blob/main/Assets/ResNet101_training_history.png?raw=true" alt="Training History of ResNet-101">
    <p><strong>Figure 3:</strong> ResNet101 – Training and Validation loss and accuracy.</p>
</div>

<div align="center">
    <img src="https://github.com/tzurbar/046211/blob/main/Assets/Dinov2_training_history.png?raw=true" alt="Training History of ResNet-101">
    <p><strong>Figure 4:</strong> DINOV2 – Training and Validation loss and accuracy.</p>
</div>

The loss and accuracy curves are improving with more epochs, showing better performance. Notably, the DINOV2 curves demonstrate greater consistency between training and validation. In contrast, the ResNet101 curves exhibit more noise and instability.

**Comparing to a scientific [paper](https://ieeexplore.ieee.org/abstract/document/10489335/authors#authors)**:

We compared our results with a study that used our dataset with various models. Our findings suggest that training full models without pre-trained fine-tuning methods is more effective. The study reported model accuracies ranging from 91.75% to 99.59%, while our model's accuracy stands at 77.3% and 73.3%.

## Conclusion <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/concept_11063232.png" alt="Icon" width="30" height="30">
- **Pretrained Models**:
We anticipated achieving better results with pretrained models through fine-tuning compared to developing a new architecture from scratch. However, our findings indicate that training full models yields superior results. Nonetheless, full model training requires substantial computational resources and time, which were constraints for our project.
- **ResNet101 VS DINOV2**:
Our comparison revealed that DINOV2 (ViT) outperforms ResNet101 (CNN) in terms of accuracy and stability when using pretrained models with fine-tuning. This disparity is likely due to DINOV2 being trained on 142 million images compared to ResNet101’s 1.4 million, marking a significant difference in training data volume.
- **Classes Accuracy**:
As demonstrated in the table and graph, accuracy improves with an increasing number of images. For precise diagnostic results in specific classes, acquiring more data related to those diseases is crucial.
## Future Work <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/part-time_12322679.png" alt="Icon" width="30" height="30">
For future improvements, we propose gathering additional data and integrating it into our models.
We also plan to explore the addition of DORA layers in the middle of the models and experiment with more hyperparameter tuning using tools like Optuna.

## References <img src="https://github.com/tzurbar/046211/blob/main/Assets/Icons/copy_1644122.png" alt="Icon" width="30" height="30">
1. **Skin Diseases Image Dataset**  
   Kaggle. [Link](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data)

2. **Enhancing Skin Disease Classification and Privacy Preservation through Federated Learning-Based Deep Learning**  
   Raj Gaurang Tiwari, Himani Maheshwari, Vinay Gautam, Ambuj Kumar Agarwal, Naresh Kumar Trivedi. 2023 International Conference on Artificial Intelligence for Innovations in Healthcare Industries (ICAIIHI). [Link](https://ieeexplore.ieee.org/abstract/document/10489335/authors#authors)

3. **DINOv2: Self-Supervised Vision Transformers**  
   Meta. [Link](https://github.com/facebookresearch/dinov2)

4. **ResNet-101 Model Reference**  
   Microsoft. (n.d.). ResNet-101. [Link](https://huggingface.co/microsoft/resnet-101)

5. **DoRA: Improving on LoRA’s Parameter-Efficient Fine-Tuning**  
   LM Po. Medium, 2024. [Link](https://medium.com/@edmond.po/exploring-dora-improving-on-loras-parameter-efficient-fine-tuning-d72edc045f64#:~:text=DoRA%20is%20a%20promising%20technique,performance%20with%20even%20fewer%20parameters.)
