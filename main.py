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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X)
pandas.set_option('display.max_columns', None)
X_Stats.describe()

y_train_Stats = pandas.DataFrame(y_train)
y_test_Stats = pandas.DataFrame(y_test)
y_train_Stats.describe()
y_test_Stats.describe()

# We can see that the data has stratified as intended.

# Convert to float Tensor
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())
y_test = torch.squeeze(torch.from_numpy(y_test).float())


# Initializing the neural network class
class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 12)
    self.fc2 = nn.Linear(12, 8)
    self.fc3 = nn.Linear(8, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return torch.sigmoid(self.fc3(x))
net = Net(X_train.shape[1])

# Loss Function
criterion = nn.BCELoss()
optimizer = SGD(net.parameters(), lr=0.1)  ## here we're creating an optimizer to train the neural network.
#This learning rate seems to be working well so far

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(1010):

    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    train_acc = calculate_accuracy(y_train, y_pred)
    y_test_pred = net(X_test)
    y_test_pred = torch.squeeze(y_test_pred)
    test_loss = criterion(y_test_pred, y_test)
    test_acc = calculate_accuracy(y_test, y_test_pred)


    if (epoch) % 10 == 0:
        print(f'For Epoch: {epoch}')
        print(f'Training loss: {round_tensor(train_loss)} Accuracy: {round_tensor(train_acc)}')
        print(f'Testing loss: {round_tensor(test_loss)} Accuracy: {round_tensor(test_acc)}')

# If test loss is less than 0.02, then break. That result is satisfactory.
    if test_loss < 0.02:
        print("Num steps: " + str(epoch))
        break

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# Creating a function to evaluate our input
def TitanicSurvived(Sex,
                    Age,
                    Sib_Spos_Abrd,
                    Par_Chil_Abrd,
                    Pclass_1,
                    Pclass_2,
                    Pclass_3,
                    Fare
                   ):
  t = torch.as_tensor([Sex,
                    Age,
                    Sib_Spos_Abrd,
                    Par_Chil_Abrd,
                    Pclass_1,
                    Pclass_2,
                    Pclass_3,
                    Fare
                       ]) \
    .float() \
    .to(device)
  output = net(t)
  return output.ge(0.5).item(), output.item()

# Updated return function to include the continuous value in addition to the classification.

# Note: Male Sex = 0, Female = 1
TitanicSurvived(Sex=0,
                Age=0,
                Sib_Spos_Abrd=0,
                Par_Chil_Abrd=0,
                Pclass_1=0,
                Pclass_2=0,
                Pclass_3=1,
                Fare=0)

# Note: Male Sex = 0, Female = 1
TitanicSurvived(Sex=1,
                Age=0,
                Sib_Spos_Abrd=0,
                Par_Chil_Abrd=0,
                Pclass_1=1,
                Pclass_2=0,
                Pclass_3=0,
                Fare=0)

# Define categories for our confusion matrix
Categories = ['Survived','Not Survived']

# Where y_test_pred > 0.5, we categorize it as 1, or else 0.
y_test_dummy = np.where(y_test_pred > 0.5,1,0)

# Creating a confusion matrix to visualize the results.
# Model Evaluation Part 2
Confusion_Matrix = confusion_matrix(y_test, y_test_dummy)
Confusion_DF = pandas.DataFrame(Confusion_Matrix, index=Categories, columns=Categories)
sns.heatmap(Confusion_DF, annot=True, fmt='g')
plt.ylabel('Observed')
plt.xlabel('Yhat')


## Let's conduct a linear regression and evaluate the coefficients.

Reg_Out = ols("Survived ~ Sex + Age + Sib_Spos_Abrd + Par_Chil_Abrd + Pclass_1 + Pclass_2 + Pclass_3 + Fare",
              data = Titanic_WP).fit()

print(Reg_Out.summary())

# Example regression output. It appears that changes in Sex have the largest effect on Survived.
# Passenger class also has a large effect on Survived. Increases in Pclass_1 have large effects on Survived.

#   =================================================================================
#                       coef    std err          t      P>|t|      [0.025      0.975]
#   ---------------------------------------------------------------------------------
#   Intercept         0.3479      0.031     11.104      0.000       0.286       0.409
#   Sex               0.5075      0.028     18.115      0.000       0.453       0.563
#   Age              -0.0062      0.001     -5.942      0.000      -0.008      -0.004
#   Sib_Spos_Abrd    -0.0502      0.013     -3.804      0.000      -0.076      -0.024
#   Par_Chil_Abrd    -0.0194      0.018     -1.072      0.284      -0.055       0.016
#   Pclass_1          0.2936      0.033      8.896      0.000       0.229       0.358
#   Pclass_2          0.1190      0.024      4.958      0.000       0.072       0.166
#   Pclass_3         -0.0647      0.018     -3.668      0.000      -0.099      -0.030
#   Fare              0.0004      0.000      1.235      0.217      -0.000       0.001
#   ==============================================================================

# The importance of Sex = 1 (Female) and Pclass_1 = 1 (First class) is reflected in the output of our
# ...TitanicSurvived function.
# Note: Male Sex = 0, Female = 1
TitanicSurvived(Sex=1,
                Age=0.2,
                Sib_Spos_Abrd=1,
                Par_Chil_Abrd=1,
                Pclass_1=1,
                Pclass_2=0,
                Pclass_3=0,
                Fare=1)

## Feel free to play around with the inputs and evaluate the output.
TitanicSurvived(Sex=0,
                Age=0,
                Sib_Spos_Abrd=1,
                Par_Chil_Abrd=1,
                Pclass_1=0,
                Pclass_2=0,
                Pclass_3=1,
                Fare=1)





# Let's create a quick K-nearest neighbors model and see what we get.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas
import numpy

Accuracy_Values = []

for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

# Calculate the accuracy of the model
    if i % 2 == 0:
    print("Iteration K =",i,"Accuracy Rate=", knn.score(X_test, y_test))
    print(knn.score(X_test, y_test))
    Accuracy_Values.append([i,knn.score(X_test, y_test)])

K_Accuracy_Pair = pandas.DataFrame(Accuracy_Values)
K_Accuracy_Pair.columns=['K','Accuracy']

# Let's see the K value where the accuracy was best:

K_Accuracy_Pair[K_Accuracy_Pair['Accuracy']==max(K_Accuracy_Pair['Accuracy'])]

# Best iteration was K = 12 and accuracy = 82.5%.
# This is actually slightly better than the neural network's accuracy.
# The neural network's accuracy was 81.46%.




# And let's do a quick logistic regression model:

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

Logit = LogisticRegression()

poly_accuracy = []

polynomials = range(1,10)

for poly_degree in polynomials:
    poly = PolynomialFeatures(degree = poly_degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    Logit.fit(X_poly, y_train)
    y_pred = Logit.predict(X_test_poly)
    print('Polynomial Degree:',poly_degree,'Accuracy:',round(Logit.score(X_test_poly, y_test),2))
    poly_accuracy.append([poly_degree,round(Logit.score(X_test_poly, y_test),2)])

Polynomial_Accuracy = pandas.DataFrame(poly_accuracy)
Polynomial_Accuracy.columns = ['Polynomial','Accuracy']

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Optimal logistic regression model has a polynomial value of 5, 6, or 9 and an accuracy rate of 83%.
# It looks like our logit model is slightly more accurate than k-nearest neighbor and neural network models.
# Accuracy with neural network: 81.5%.
# Accuracy with logistic regression: 83%.
# Accuracy with k-nearest neighbors: 82.5%.


