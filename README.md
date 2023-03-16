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

Text

---
## 2. OBJECTIVES
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
