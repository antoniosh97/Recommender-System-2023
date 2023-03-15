# RECOMMENDER SYSTEM
Final project for the 2022-2023 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by **Evaristo Broullon**, **Joan Dayas**, **Brenda Fernández** and **Antonio Sánchez**. 


Advised by **Paula Gómez Duran**.

Table of Contents
=================

- [RECOMMENDER SYSTEM](#recommender-system)
- [Table of Contents](#table-of-contents)
	- [INTRODUCTION AND MOTIVATION](#introduction-and-motivation)
	- [DATASETS](#datasets)
		- [Amazon Instruments](#amazon-instruments)
		- [MovieLens](#movielens)
	- [ARCHITECTURE AND RESULTS](#architecture-and-results)
		- [Splitting Datasets](#splitting-datasets)
		- [Pipeline functions](#pipeline-functions)
			- [Training](#training)
			- [Inference](#inference)
		- [Pipeline](#pipeline)
	- [ABLATION STUDIES](#ablation-studies)
	- [STATE OF ART](#state-of-art)
		- [Collaborative Recommender System](#collaborative-recommender-system)
		- [Cold Start](#cold-start)
		- [Contextual Bandits](#contextual-bandits)
		- [Leave-One-Out error](#leave-one-out-error)
		- [Paper: Full ranking in test](#paper-full-ranking-in-test)
		- [Paper: Sampling Techniques](#paper-sampling-techniques)
		- [Conclusions](#conclusions)
	- [MODELS](#models)
		- [FACTORIZATION MACHINE](#factorization-machine)
		- [NEURAL COLLABORATIVE FILTERING](#neural-collaborative-filtering)
		- [POPULARITY-BASED](#popularity-based)
		- [RANDOM](#random)
		- [Final results](#final-results)
	- [HYPERPARAMETER TUNING](#hyperparameter-tuning)
		- [FACTORIZATION MACHINE](#factorization-machine-1)
		- [NEURAL COLLABORATIVE FILTERING](#neural-collaborative-filtering-1)
	- [END TO END SYSTEM](#end-to-end-system)
	- [HOW TO](#how-to)
		- [HOW TO PREPARE THE DATASET FROM SCRATCH](#how-to-prepare-the-dataset-from-scratch)
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
				- [RabbitMQ](#rabbitmq)
				- [Video processor app](#video-processor-app)
		- [HOW TO RUN THE PROGRAM - video\_capture](#how-to-run-the-program---video_capture)
	- [END](#end)
---
---


// ![Image](Management/_images/nn.png)
## INTRODUCTION AND MOTIVATION

Text


## DATASETS
### Amazon Instruments
### MovieLens
## ARCHITECTURE AND RESULTS
### Splitting Datasets
### Pipeline functions
#### Training
#### Inference
### Pipeline
## ABLATION STUDIES
## STATE OF ART
### Collaborative Recommender System
### Cold Start
### Contextual Bandits
### Leave-One-Out error
### Paper: Full ranking in test
### Paper: Sampling Techniques
### Conclusions
## MODELS
### FACTORIZATION MACHINE
### NEURAL COLLABORATIVE FILTERING
### POPULARITY-BASED
### RANDOM
### Final results
## HYPERPARAMETER TUNING
### FACTORIZATION MACHINE
### NEURAL COLLABORATIVE FILTERING
## END TO END SYSTEM
## HOW TO
### HOW TO PREPARE THE DATASET FROM SCRATCH
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
##### RabbitMQ
##### Video processor app
### HOW TO RUN THE PROGRAM - video_capture
## END

<!-- 

## ARCHITECTURE AND RESULTS
Text
### Splitting Datasets
Text

#### Pipeline functions
Text
### Pipeline
### MODEL IMPROVEMENTS

#### FIRST APPROACH
Text

#### SECOND APPROACH

Text


#### THIRD APPROACH

Text


#### Final results

Text



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
