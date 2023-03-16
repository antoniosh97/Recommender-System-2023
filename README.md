# RECOMMENDER SYSTEM PROJECT
Final project for the 2022-2023 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Antonio Sánchez**, **Brenda Fernández**, **Evaristo Broullon** and **Joan Dayas**. 


Advised by **Paula Gómez Duran**.

Table of Contents
=================

- [RECOMMENDER SYSTEM PROJECT](#recommender-system-project)
- [Table of Contents](#table-of-contents)
	- [1. INTRODUCTION AND MOTIVATION](#1-introduction-and-motivation)
	- [2. OBJECTIVES](#2-objectives)
	- [3. STATE OF ART](#3-state-of-art)
			- [3.1 COLLABORATIVE RECOMMENDATION SYSTEM](#31-collaborative-recommendation-system)
			- [3.2 MODELS](#32-models)
				- [Factorization Machine](#factorization-machine)
				- [Neural Collaborative Filtering](#neural-collaborative-filtering)
				- [Popularity Based](#popularity-based)
				- [Random](#random)
			- [3.3 COLD START](#33-cold-start)
			- [3.4 LEAVE-ONE-OUT](#34-leave-one-out)
			- [3.5 PAPER: FULL RANKING](#35-paper-full-ranking)
			- [3.6 PAPER: SAMPLING TECHIQUES](#36-paper-sampling-techiques)
	- [4. METHODOLOGY](#4-methodology)
		- [4.1 DATASETS](#41-datasets)
			- [Amazon Instruments](#amazon-instruments)
			- [MovieLens](#movielens)
		- [4.2 ARCHITECTURE](#42-architecture)
			- [Splitting Datasets](#splitting-datasets)
			- [Pipeline functions](#pipeline-functions)
				- [Training](#training)
				- [Inference](#inference)
			- [Pipeline](#pipeline)
	- [5. ABLATION STUDY](#5-ablation-study)
		- [5.1 EXPERIMENT 1:](#51-experiment-1)
			- [Experiment setup](#experiment-setup)
			- [Results](#results)
			- [Conclusions](#conclusions)
		- [5.2 EXPERIMENT 2:](#52-experiment-2)
			- [Experiment setup](#experiment-setup-1)
			- [Results](#results-1)
			- [Conclusions](#conclusions-1)
		- [5.3 FINAL RESULTS](#53-final-results)
		- [5.4 CONCLUSIONS](#54-conclusions)
	- [6. DEVELOPMENT](#6-development)
		- [PREPARE THE DATASET FROM SCRATCH](#prepare-the-dataset-from-scratch)
			- [Download Dataset](#download-dataset)
			- [Clean dataset](#clean-dataset)
		- [HOW TO EXTRACT OPTICAL FLOW](#how-to-extract-optical-flow)
		- [HOW TO EXTRACT FEATURES FROM VIDEOS](#how-to-extract-features-from-videos)
		- [HOW TO TRAIN THE MODEL](#how-to-train-the-model)
			- [Setting the environment in Google Drive](#setting-the-environment-in-google-drive)
			- [Running training scripts](#running-training-scripts)
		- [HOW TO RUN THE PROGRAM - video\_processor](#how-to-run-the-program---video_processor)
			- [Installation](#installation)
				- [Install Docker](#install-docker)
				- [Install docker-compose](#install-docker-compose)
				- [Install Miniconda](#install-miniconda)
				- [Create your Miniconda environment](#create-your-miniconda-environment)
				- [Create your .env file](#create-your-env-file)
			- [RUN the project](#run-the-project)
				- [??](#)
	- [END](#end)

---
---


// ![Image](Management/_images/nn.png)
## 1. INTRODUCTION AND MOTIVATION

<div align="justify">

Deep Learning is one of the next big things in Recommendation Systems technology. The exponential growth due to the interest in implementing recommendations that fit the interests of the user.

It is worth noting several sectors that have suffered a greater impact: social networks are conditioned by the user's interest in publications, online commerce has the purpose of showing products that satisfy the user's needs and finally the marketing sector that with information of their clients to define the most optimal marketing action.

In general, recommender systems are algorithms based on suggesting relevant items to users.

Among the various reasons why this field is interesting to study, we found that improving the user experience encourages them to discover new products of interest. In addition, in case the recommendations are accurate, the commitment with the application is strengthened and this causes a win-win since the company captures data to improve its marketing strategies.
</div>

---
## 2. OBJECTIVES
<div align="justify">

In middle of such a variety of actual options in the market for recommender systems, the different problems, complexity and techniques used for data sampling for example, our ambition with this project is to know and understand the advantages of the main different models used today by the industry, sampling techniques and metrics to evaluate the end results of each model tested.

Due to the maturity of the existent technology, vastly used nowadays, our anchor to understand this type of system will be the FM - Factorization Machina and then extending to other models like Random, Popularity and the NCF - Neural Collaborative Filtering.

From the beginning, we could notice and understand the complexity of recommending something to a customer. There are a lot of variables involved, from the personal taste and style to a general country culture. We understand the importance of using all those context content to recognize the personal taste of a customer, but with it, we come across with a very big complexity introducing those variables in our project. Not only by the scarcity of data resources from the datasets available, but also by time to work with a context based recommender system. This is the main reason we are going to explore the “implicit feedback”, considering as positive an interaction of the customer with a product. For the negative interactions, we will need to generate it using a “negative sampling” technique commonly used by this industry. 

In a first superficial look, a recommender system can demonstrate help customers to get close with the products of their interest, but also can help companies to get users more engaged with their brand. On the other hand, instead of adopting a user-centered approach focusing on preference prediction, they shape user preferences and guide choices. This impact is significant and deserves ethical attention. 

</div>

---
## 3. STATE OF ART
#### 3.1 COLLABORATIVE RECOMMENDATION SYSTEM
#### 3.2 MODELS
##### Factorization Machine
##### Neural Collaborative Filtering
##### Popularity Based
##### Random
#### 3.3 COLD START
#### 3.4 LEAVE-ONE-OUT
#### 3.5 PAPER: FULL RANKING
#### 3.6 PAPER: SAMPLING TECHIQUES

<!-- ### HYPERPARAMETER TUNING
#### FACTORIZATION MACHINE
#### NEURAL COLLABORATIVE FILTERING -->

---
## 4. METHODOLOGY 
### 4.1 DATASETS
#### Amazon Instruments
#### MovieLens
### 4.2 ARCHITECTURE
#### Splitting Datasets
#### Pipeline functions
##### Training
##### Inference
#### Pipeline
---
## 5. ABLATION STUDY
### 5.1 EXPERIMENT 1: 
#### Experiment setup
#### Results
#### Conclusions
### 5.2 EXPERIMENT 2: 
#### Experiment setup
#### Results
#### Conclusions
### 5.3 FINAL RESULTS
### 5.4 CONCLUSIONS
---
## 6. DEVELOPMENT
### PREPARE THE DATASET FROM SCRATCH
#### Download Dataset
#### Clean dataset
### HOW TO EXTRACT OPTICAL FLOW
### HOW TO EXTRACT FEATURES FROM VIDEOS
### HOW TO TRAIN THE MODEL
#### Setting the environment in Google Drive
#### Running training scripts
### HOW TO RUN THE PROGRAM - video_processor
#### Installation
##### Install Docker
##### Install docker-compose
##### Install Miniconda
##### Create your Miniconda environment
##### Create your .env file
#### RUN the project
##### ??
## END

<!-- 


### HOW TO TRAIN THE MODEL

#### Setting the environment

Text

#### Running training scripts

Text

### HOW TO RUN THE PROGRAM

#### Installation

Text

##### Install Miniconda

To test your installation, in your terminal window or Anaconda Prompt, run the command: 
```bash
$ conda list
```
And you should obtain the list of packages installed in your base environment.

##### Create your Miniconda environment

>  Notice: This repository has been designed in order to allow being executed with two different environments. If you have a GPU in your computer, make use of the file "environment_gpu.yml" during the next section, in case you only have CPU, use the file "environment.yml" to prepare your environment.

Execute:

```bash
 $ conda env create -f environment_gpu.yml
```

This will generate the `videoprocgpu` environment with all the required tools and packages installed.

Once your environment had been created, activate it by typing:

```bash
$ conda activate videoprocgpu
```


##### RUN the project

Text


### HOW TO RUN THE PROGRAM

Text    -->
