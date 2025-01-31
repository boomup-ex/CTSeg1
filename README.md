## Automated Segmentation and Morphological Measurement of Carpal Tunnel Tissues using Edge-Guided Dual-Stream Network


## Dataset
We constructed a wrist MRI database to test the effectiveness of the proposed wrist canal tissue segmentation model.
Our wrist canal MRI data is sourced from the Radiology Department of Shanxi Provincial People's Hospital. 
To protect patient privacy, we only provide. npy files that do not contain header files containing patient personal information.
The data is stored in the data folder, and the .zip file in the folder can be decompressed for model training

## Note
Our research has been summarized and submitted to 《The Visual Computer》 as an article titled:"Automated Segmentation and Morphological Measurement of Carpal Tunnel Tissues using Edge-Guided Dual-Stream Network"

## 1、Getting Start
```git clone https://github.com/boomup-ex/CTSNet.git```
    

## 2、Requirements
```
mmcv==1.3.8
nibabel==4.0.2
numpy==1.21.6
opencv-contrib-python==4.7.0.72
opencv-python==4.7.0.72
opencv-python-headless==4.8.1.78
scikit-image==0.19.3
scikit-learn==0.22.2
scipy==1.7.3
timm==0.5.4
torch==1.11.0+cu113
torchvision==0.12.0+cu113
```
Install all dependent libraries:
```
pip install -r requirements.txt
```

## 3、Run
```
python main.py
```
## 4、Results

### Training Curve
![train](https://github.com/user-attachments/assets/b4323ca8-1347-47f3-854e-f5005fb46eac)


### Compared with the performance of some networks
![myplot](https://github.com/user-attachments/assets/f7df75fa-2d56-4431-ae42-5c876a8d8b0d)

### Ablation study
Note: A:Dual stream encoder B:Medium scale feature extraction module C:Edge steering module D:Customised loss function
![bat](https://github.com/user-attachments/assets/9ab8d026-42e7-4067-91a7-c3ba5fcbb153)


### Framework of the proposed model
![image](https://github.com/user-attachments/assets/0c385c5d-e565-4a92-8cd7-171d34d830a9)


### Segmentation visualization results of the MRI carpal tunnel, The first column shows the original MRI, the second through fifth columns show the segmentation results of other state-of-the-art methods, the sixth shows the segmentation results of proposed method in this paper, the last column shows the manual segmentation by the clinician.
![image](https://github.com/user-attachments/assets/e17cf851-ca2d-48fb-a25c-40a3fcfa9451)






![image](https://github.com/user-attachments/assets/87ce8fd0-81ee-4df5-b662-a77ccfca0993)

Test results on the dataset used for evaluation. The proposed method continues to demonstrate robust segmentation outcomes for both modalities across additional datasets, substantiating its exceptional generalisation capabilities.
