# end-to-end_mazie_disease_classification

## Project Background and References

This project addresses the challenge of **identifying maize leaf diseases** in complex natural environments, particularly in Africa, where diseases such as **Maize Lethal Necrosis** and **Maize Streak Virus** cause major yield losses for smallholder farmers.

### Dataset

The dataset used in this project contains **18,148 curated images** of maize leaves (both healthy and diseased), captured using **smartphone cameras** in **Tanzania**. It is the **largest publicly available dataset** for maize leaf health classification and is suitable for tasks such as:

- Disease diagnosis via image classification
- Object detection and segmentation
- Real-time field-based prediction systems

> **Citation**:  
> Maize is one of the most important staple food and cash crops that are largely produced by majority of smallholder farmers throughout the humid and sub-humid tropics of Africa. This dataset contains images of maize leaves collected in Tanzania with the aim to support the early diagnosis of diseases and contribute to improved food security in Africa.  
> **Total images:** 18,148  
> *(Source: Maize disease dataset collected in Tanzania, 2021)*

---

### Model: LFMNet Architecture

The model implemented is **LFMNet**, a lightweight multi-attention convolutional neural network architecture designed to tackle challenges like **background interference**, **high inter-class similarity**, and **real-time inference**.

#### Reference

> **Hu Jian, Jiang Xinhua, Gao Julin, Yu Xiaofang**  
> *LFMNet: a lightweight model for identifying leaf diseases of maize with high similarity*  
> Frontiers in Plant Science, Volume 15, 2024  
> [DOI: 10.3389/fpls.2024.1368697](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1368697)

#### Summary of LFMNet Layers:

| Layer         | Output Shape     | Description                                                                
|---------------|------------------|----------------------------------------------------------
| Conv1         | 24×112×112       | 7×7 Conv, stride=2, BN, ReLU                                               
| PMFFM1        | 24×56×56         | Partial Conv with dilation, 3×3 kernel, BN, ReLU                            
| MAttion1      | 48×28×28         | MSA + PPA + convolutional layers                                            
| PMFFM2        | 48×28×28         | Partial Conv with dilation, 1×1 kernel                               
| MAttion2      | 96×14×14         | Same as MAttion1 but deeper features                                        
| MAttion3      | 192×7×7          | Expanded features, multi-branch pooling                                     
| MAttion4      | 256×3×3          | Final multi-attention fusion                                                
| Average Pool  | 256              | Global average pooling                                                      
| Classifier    | 4               | Fully connected layer                                                        


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/AbdulRasheed6/end-to-end_mazie_disease_classification
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n cnncls python=3.9.12 -y
```

```bash
conda activate cnncls
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: *************.dkr.ecr.us-east-1.amazonaws.com/maize

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  *********************.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




# AZURE-CICD-Deployment-with-Github-Actions

## Save pass:

**********************************


## Run from terminal:

docker build -t maizedisease.azurecr.io/maize:latest .

docker login maizediseaseapp.azurecr.io

docker push maizediseaseapp.azurecr.io/maize:latest


## Deployment Steps:

1. Build the Docker image of the Source Code
2. Push the Docker image to Container Registry
3. Launch the Web App Server in Azure 
4. Pull the Docker image from the container registry to Web App server and run 
