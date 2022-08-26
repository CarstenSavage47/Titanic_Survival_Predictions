import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  ## optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from statsmodels.formula.api import ols

import pandas
import numpy as np

Titanic = pandas.read_csv('/Users/carstenjuliansavage/Desktop/titanic.csv')

# Reorder vars for more logical order
Titanic = Titanic[['Name','Sex','Age','Siblings/Spouses Aboard','Parents/Children Aboard','Pclass','Fare','Survived']]

Titanic.columns = ['Name','Sex','Age','Sib_Spos_Abrd','Par_Chil_Abrd','Pclass','Fare','Survived']

# Add prefix to the newly-created dummy vars so we know what they are.
Just_Dummies = pandas.get_dummies(Titanic['Pclass'],prefix='Pclass')

Titanic_WP = pandas.concat([Titanic, Just_Dummies], axis=1)

Titanic_WP['Sex'] = np.where(Titanic_WP['Sex']=='female',1,0)

X = Titanic_WP[['Sex',
                'Age',
                'Sib_Spos_Abrd',
                'Par_Chil_Abrd','Pclass','Fare']]
y = Titanic_WP[['Survived']]