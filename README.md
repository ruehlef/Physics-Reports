# Repository for "Data science in string theory"

In this repository, you can find material used in the Physics Report book "Data science in string theory". It is meant to get you started by providing simple example codes for the various data science techniques introduced in the book.

Feel free to use and modify the code at will. If you find this useful, you could consider [citing](./bibliography.bib) the book in your work. 

# Content
* [Chapter 2](./Chapter_2) introduces common neural network layers.
  * In [Chapter 2.1](./Chapter_2/2.1) we implement an extremely simple fully connected feed-forward neural network. It solves the binary classification task introduced in Chapter 2.1. We implement the network and the training in different languages to illustrate how to use them:
	  - Keras
	  - Mathematica
	  - PyTorch
	  - TensorFlow
  
  * In [Chapter 2.4](./Chapter_2/2.4), we give examples and illustrations for CNNs with different kernel sizes and strides:
	  - convolutions 
	  - max pooling
  
* [Chapter 3](./Chapter_3) illustrates over- and underfitting, as well as the regularizing effect of dropout layers and weight decay.

* [Chapter 5](./Chapter_5/GA_Example.htm) illustrates the extremization of a simple 2D function using **Genetic Algorithms**.

* [Chapter 6](./Chapter_6) illustrates the  computation of persistent homology of a point cloud that lies on a cylinder surface in JAVAPLEX via [Jython](./Chapter_6/Jython) and [JAVA](./Chapter_6/JAVA)

* [Chapter 7](./Chapter_7) introduces several codes related to **unsupervised learning techniques** beyond neural networks.
  * In [Chapter 7.2](./Chapter_7/7.2) we introduce **Voronoi diagrams** which often appear in distance-based clustering.
  * In [Chapter 7.3](./Chapter_7/7.3) we provide the Mathematica code used to illustrate **Principal Component Analysis**
  * In [Chapter 7.4](./Chapter_7/7.4) we provide the Python code to perform **K-Means Clustering** with Scikit learn
  * In [Chapter 7.5](./Chapter_7/7.5) we provide a Mathematica implementation of **Mean Shift Clustering**.
  * In [Chapter 7.6](./Chapter_7/7.6) we provide the Python code to perform **Gaussian Expectation-Maximization Clustering** with Scikit learn
  * In [Chapter 7.9](./Chapter_7/7.9) we provide the Python code used to compare the different clustering algorithms discussed in this chapter.

* [Chapter 8](./Chapter_8) introduces several codes related to **Reinforcement learning**. We provide an example for both tabular and deep RL.
  * In [Chapter 8 SARSA](./Chapter_8/SARSA) we introduce a tabular method to solve a maze using the SARSA algorithm. This example is discussed in Chapter 8.5 of the book.
  * In [Chapter 8 DEEP_RL](./Chapter_8/DEEP_RL) we illustrate the **A3C** and **DQN** algorithm discussed in Chapter 8.6 of the book by training an agent to play Tic Tac Toe. This is meant to illustrate the algorithms and to show how to use Chainerrl together with the gym environment to train RL agents. You can then play the game against the trained agent to see how it behaves, which tactics it learned/didn't learn, which moves it was considering, etc.
      - [A3C](./Chapter_8/DEEP_RL/A3C) contains the A3C implementation
	  - [DQN](./Chapter_8/DEEP_RL/DQN) contains the DQN implementation
	
* [Chapter 9](./Chapter_9) introduces several codes related to **supervised learning techniques** beyond neural networks.
  * In [Chapter 9.2](./Chapter_9/9.2) we introduce various **Decision Trees** and **Random Forest** implementations that can be used in clustering or regression
  * In [Chapter 9.3](./Chapter_9/9.3) we introduce various **Support Vector Machines**for clustering or regression
  
  
# How to get started  with the code
Most of the examples use Python. Chances are you already have (at least) one Python environment on your system. If not, install one (e.g. Miniconda or Anaconda). I used python3 in all my examples. I think they should also work with python2, but I have not tested them. Since python2 support runs out end of 2019, it is probably better to start with python3 anyways.

SOme examples also illustrate techniques with mathematica, and for the Persistent Homology Code, you'llneed Jython or JAVA.

### Create a virtual python environment
It is best to create a virtual environment and install the packages there, in order to not interfere with the installation you already have, and to be able to use different versions of packages for different projects. Setting up a virtual environment is much easier than it sounds. If you do not want to do this, you can skip it, but it might actually be more difficult to install the packages you need without a virtual environment.

To set up a virtual environment using an existing python installation, follow [these instructions](https://docs.python.org/3/library/venv.html). The steps are really simple. On Linux/Mac do the following:

1. Create the virtual environment. In a shell enter 
```
python3 -m venv ~/my_venv
``` 
where `~/my_venv` is where the new environment shall be stored (feel free to use anything else).

2. Activate the virtual environment. To activate it enter 
```
source ~/my_venv/bin/activate
```

3. Do whatever you want to do, e.g. `pip install numpy` or `python hello_world.py`.

4. Deactivate the virtual environment. This is optional (you could simply close the terminal), but if you want to return to the normal environment simply enter `deactivate`.

### Install/upgrade python packages using pip
Next, you probably need to install some packages/libraries. To do so, use the command

```
pip install --user xxx
```

where `xxx` is the package you want to install. The `--user` flag means that the package will be installed into your local (home) folder rather than system-wide.
You can also upgrade existing packages using

```
pip install --user --upgrade xxx
```

Some packages you might want to install are

* `numpy`
* `scipy`
* `sympy`
* `matplotlib`
* `pandas`
* `seaborn`
* `tkinter`
* `sklearn`
* `chainerrl`
* `torch`
* `keras`
* `tensorflow`
