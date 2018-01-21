# Introduction to Machine learning 

This is the main repository of the Machine learning course teach at November/December of 2017. You can find different folder containing the dataset and content needed for the course.

## Table of content

* [Slides](#slides)
* [Getting Started](#getting-started)
  * [Pre Setup](#pre-setup)
  * [Python Installation](#python-installation)
  * [Environment Setup](#environment-setup)
* [Interesting links](#interesting-links)
* [Questions](#questions)
* [Previous courses](#courses)

## Slides

| Class | Slides | Solutions |
| ------ | ------ | ------ |
| Session 1 - The What Why and when of Machine Learning | [Slides](https://docs.google.com/presentation/d/13au8KeMavSuW1GjYMbFT2rWlb6EVjEgINNsbdidjoFE/edit?usp=sharing) | [Solutions](https://github.com/altran-machine-learning-course/course_11_2017/blob/master/Week1/Session1_solutions.ipynb) |
| Session 2 - Feature Engineering | [Slides](https://docs.google.com/presentation/d/1W9ZADTIKhLyJXycjbqPPzLq8nySfGcOgDH7wNwfnGfY/edit?usp=sharing) | [Solutions](https://github.com/altran-machine-learning-course/course_11_2017/blob/master/Week1/Session2_solutions.ipynb) |
| Session 3 - Linear Classifiers | [Slides](https://docs.google.com/presentation/d/1el2enYn7nMsDVX5JMpHZW4EzsknJ0ML7onqJLyvPIGc/edit?usp=sharing) | |
| Session 4 - Classifier Optimization | [Slides](https://drive.google.com/open?id=1c3idr6asSKBdmE_Y0jToBkSg7BuNDkG4b65cmLHEBt8) | |
| Session 5 - Neural Network Overview | [Slides](https://drive.google.com/open?id=1leQ8uL-Cq7zWFABZOlLkFJUJYn0pccbCuCUwnOPsuoo) | |


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

```
Cython==0.26.1
ipykernel==4.6.1
ipython==6.1.0
ipython-genutils==0.2.0
ipywidgets==7.0.0
jupyter-client==5.1.0
jupyter-console==5.2.0
jupyter-core==4.3.0
jupyterlab==0.27.0
jupyterlab-launcher==0.4.0
matplotlib==2.0.2
notebook==5.0.0
numpy==1.13.1
numpydoc==0.7.0
pandas==0.20.3
pandocfilters==1.4.2
scikit-image==0.13.0
scikit-learn==0.19.0
scipy==0.19.1
seaborn==0.8
tensorflow==1.4.0
tensorflow-tensorboard==0.4.0rc3
```

## Interesting links
### Session 1
* [Common myths in ML](http://www.iamwire.com/2017/07/3-common-myths-around-machine-learning/156129) - Good article.
* [Matplotlib cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf) - Cheatsheet for using Python's matplotlib library.
* [Pandas cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf) - Cheatsheet for using Python's pandas library.
* [Pandas cookbook](https://github.com/jvns/pandas-cookbook) - Recipes for using Python's pandas library
* [Seaborn cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf) - Cheatsheet for using Python's seaborn library.
* [Two Minute Papers](https://www.youtube.com/user/keeroyz/videos) - Youtube channel with interesting ML videos.
* [Violin Plots](https://blog.modeanalytics.com/violin-plot-examples/) - Description and examples of use.
### Session 2
* [Dataset](https://machinelearningmastery.com/difference-test-validation-datasets/) - Differences between training, test and validation dataset.
* [Feature engineering](https://elitedatascience.com/feature-engineering-best-practices) Tips on feature engineering
* [Feature selection](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/) - Introduction to feature selection methods.
* [Pearson's correlation](https://blog.bigml.com/2015/09/21/looking-for-connections-in-your-data-correlation-coefficients/) - Simple explanation of the pearson's correlation method.
* [PCA](http://setosa.io/ev/principal-component-analysis/) - PCA Algoritm info
* [PCA2](https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/) - More PCA algorithm info.
### Session 3
* [Random Forest](https://www.youtube.com/watch?v=D_2LkhMJcfY) - Good video for understand the algorithm.
* [Random Forest 2](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/ ) - Random forest algorithm  introduction with explanations of the main concepts.
* [SVM](https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html) - SVM algorithm simple explanation.
* [SVM 2](https://www.svm-tutorial.com/2014/11/svm-understanding-math-part-1/) - SVM algorithm advanced explanation.
* [SVM 3](http://cs.stanford.edu/people/karpathy/svmjs/demo/) - Demo.
* [Visual Machine Learnign](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) - A visual introduction to machine learning.
### Session 4
* [Ensamble methods](https://www.toptal.com/machine-learning/ensemble-methods-machine-learning) - Description of the ensambled methods.
* [Sklearn](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/) - Examples of use of this tool.

## Questions

If you want to ask something, feel free to write your question in the issues section.
 
## Courses
* November '17 (12 assistants)
* January '18
