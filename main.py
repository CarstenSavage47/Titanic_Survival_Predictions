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
                'Par_Chil_Abrd',
                'Pclass_1',
                'Pclass_2',
                'Pclass_3',
                'Fare']]
y = Titanic_WP[['Survived']]


# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y)

# Split dataframe into training and testing data. Remember to set a seed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X)
pandas.set_option('display.max_columns', None)
X_Stats.describe()

# Convert to float Tensor
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())
y_test = torch.squeeze(torch.from_numpy(y_test).float())


