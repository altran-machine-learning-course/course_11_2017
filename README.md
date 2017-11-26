# Introduction to Machine learning 

This is the main repository of the Machine learning course teach at November/December of 2017. You can find different folder containing the dataset and content needed for the course.

## Table of content

* [Slides](#slides)
* [Getting Started](#getting-started)
  * [Pre Setup](#pre-setup)
  * [Python Installation](#python-installation)
  * [Environment Setup](#environment-setup)
* [Questions](#questions)
* [Changelog](#changelog)

## Slides

| Class | Slides |
| ------ | ------ |
| Week 1 - The What Why and when of Machine Learning | [Slides](https://docs.google.com/presentation/d/13au8KeMavSuW1GjYMbFT2rWlb6EVjEgINNsbdidjoFE/edit?usp=sharing) |
| Week 2 - Feature Engineering | [Slides](https://docs.google.com/presentation/d/1W9ZADTIKhLyJXycjbqPPzLq8nySfGcOgDH7wNwfnGfY/edit?usp=sharing)
| Week 3 - Linear Classifiers | [Slides](https://docs.google.com/presentation/d/1el2enYn7nMsDVX5JMpHZW4EzsknJ0ML7onqJLyvPIGc/edit?usp=sharing)

## Interesting links
PCA: [http://setosa.io/ev/principal-component-analysis/](http://setosa.io/ev/principal-component-analysis/)


## Getting Started

### Pre Setup
* Make an account on github.com
  * Does not have to be your regular account. You could also make a temporary one just for this class/project
  * You will use this account for all your projects
  * You will have the option to set up a landing web page to publish your project beautifully as well

* **Send us your github ID**
  * This is very important, as we will be assigning the teams according to your IDs
  * Also send us your knowledge level in Machine learning (Beginner/Intermediate/Expert)

* If you are new to git and/or github and will be using it from windows
  * Download GitHub Desktop (https://desktop.github.com/ )
  * Go through a couple of simple git tutorials understanding the purpose of git (https://guides.github.com/activities/hello-world/)

* If you're familiar with python, you can skip the Python setup section
  * If not, I'd suggest you install python the same way mentioned


### Python Installation
#### Windows Installation

* For windows, the easiest thing to do is to install a distribution of python, and not just the raw python installation. 

* One of the most popular distributions of Python is Anaconda (https://www.anaconda.com/download/)

* Download the 3.6 version (because more future compatibility)
  * For our purposes, we don't care as long as version is greater that 2.7

* Follow Installation instructions
  * Add it to your path, as that would make life so much easier.
  * Do it during the installation ![installer](https://i.imgur.com/QcMBDZ5.png)
  * or if you know what you're doing, set it later. [Not Recommended]

#### Linux Installation

* Assuming that you're an expert
  * `sudo apt-get install python3.6`
  * `sudo apt-get install jupyter-notebook python-scipy python-spyder`


### Environment Setup 
#### Anaconda (Windows)
  * Open up Anaconda Navigator
  * If it asks you to make a virtual environment, do that with the default settings
  * Open up an anaconda prompt ![anaconda-prompt](https://i.imgur.com/bzQpBx8.png)
  * install the required packages `conda install seaborn scikit-learn matplotlib`
  *// [Optional]// if you like a scientific IDE - ` conda install spyder `


#### Linux
  * if you do not have a virtual-env : `sudo pip install matplotlib seaborn scikit-learn scipy numpy notebook`
  * if you do have a virtual env, skip the sudo

* **Cheers! You're ready to go! Open up spyder or notebook**

## Questions

If you want to ask something, feel free to write your question in the issues section.
 
## Changelog
07/11/2017 First version
