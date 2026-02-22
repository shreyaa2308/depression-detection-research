# -*- coding: utf-8 -*-


<h1><center>INTENTION MODEL

<h2><center> RANDOM FOREST
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('intent_mod.csv')

hash_bag = {}

# split data
X_train, X_test, y_train, y_test = train_test_split(df['final_status'].fillna(" "), df['speech_act'], test_size=0.2, random_state=42)

# convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
ac  = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
hero = ConfusionMatrixDisplay(con).plot()
plt.show()
print("ACCURACY ----> " , ac)
print("CLASSFICATION REPORT \n")
print(report)
rreport_list =  report.splitlines()

key = ['parameter','precision','recall','f1-score','support']
speech_tags = ['assertive','commissive','directive','expressive']
hash_bag["r_a_lis"] = rreport_list[2].split()
hash_bag["r_c_lis"] = rreport_list[3].split()
hash_bag["r_d_lis"] = rreport_list[4].split()
hash_bag["r_e_lis"] = rreport_list[5].split()



"""<h2><center>SVM"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('intent_mod.csv')
df.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(df['final_status'].fillna(" "), df['speech_act'], test_size=0.2, random_state=42)

# convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train random forest model
rf = SVC(kernel='linear')
rf.fit(X_train, y_train)

rf.score(X_train,y_train)

rf.score(X_test,y_test)

# make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
ac  = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
hero = ConfusionMatrixDisplay(con).plot()
plt.show()
print("ACCURACY ----> " , ac)
print("CLASSFICATION REPORT \n")
print(report)
sreport_list = report.splitlines()

key = ['parameter','precision','recall','f1-score','support']
speech_tags = ['assertive','commissive','directive','expressive']
hash_bag["s_a_lis"] = sreport_list[2].split()
hash_bag["s_c_lis"] = sreport_list[3].split()
hash_bag["s_d_lis"] = sreport_list[4].split()
hash_bag["s_e_lis"] = sreport_list[5].split()

"""<h2><center>DECISION TREE"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('shi.csv')
df.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(df['final_status'].fillna(" "), df['speech_act'], test_size=0.2, random_state=42)

# convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train random forest model
rf = DecisionTreeClassifier(max_depth=10,random_state=0)
rf.fit(X_train, y_train)

rf.score(X_train,y_train)

rf.score(X_test,y_test)

# make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
ac  = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
hero = ConfusionMatrixDisplay(con).plot()
plt.show()
print("ACCURACY ----> " , ac)
print("CLASSFICATION REPORT \n")
print(report)
dreport_list = report.splitlines()

key = ['parameter','precision','recall','f1-score','support']
speech_tags = ['assertive','commissive','directive','expressive']
hash_bag["d_a_lis"] = dreport_list[2].split()
hash_bag["d_c_lis"] = dreport_list[3].split()
hash_bag["d_d_lis"] = dreport_list[4].split()
hash_bag["d_e_lis"] = dreport_list[5].split()

"""<h1><center>KNN"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('shi.csv')
df.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(df['final_status'].fillna(" "), df['speech_act'], test_size=0.2, random_state=42)

# convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train random forest model
rf = KNeighborsClassifier(n_neighbors=3)
rf.fit(X_train, y_train)

rf.score(X_train,y_train)

rf.score(X_test,y_test)

# make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
ac  = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
hero = ConfusionMatrixDisplay(con).plot()
plt.show()
print("ACCURACY ----> " , ac)
print("CLASSFICATION REPORT \n")
print(report)
kreport_list = report.splitlines()

key = ['parameter','precision','recall','f1-score','support']
speech_tags = ['assertive','commissive','directive','expressive']

hash_bag["k_a_lis"] = kreport_list[2].split()
hash_bag["k_c_lis"] = kreport_list[3].split()
hash_bag["k_d_lis"] = kreport_list[4].split()
hash_bag["k_e_lis"] = kreport_list[5].split()

"""<h1><center>Baseline"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv('shi.csv')
df.head()

# split data
X_train, X_test, y_train, y_test = train_test_split(df['final_status'].fillna(" "), df['speech_act'], test_size=0.2, random_state=42)

# convert text data into numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# train baseline model
rf = DummyClassifier(strategy='most_frequent')
rf.fit(X_train, y_train)

rf.score(X_train,y_train)

rf.score(X_test,y_test)

# make predictions and evaluate accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score,confusion_matrix,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
ac  = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test,y_pred)
hero = ConfusionMatrixDisplay(con).plot()
plt.show()
print("ACCURACY ----> " , ac)
print("CLASSFICATION REPORT \n")
print(report)
breport_list = report.splitlines()

key = ['parameter','precision','recall','f1-score','support']
speech_tags = ['assertive','commissive','directive','expressive']

hash_bag["b_a_lis"] = breport_list[2].split()
hash_bag["b_c_lis"] = breport_list[3].split()
hash_bag["b_d_lis"] = breport_list[4].split()
hash_bag["b_e_lis"] = breport_list[5].split()

hash_bag['d_c_lis']

"""<h1> <center> VISUALISATION

<h2><center>RECALL

<h3>ASSERTIVE
"""

models  = ["b","d","k","r","s"]
val = []
aval = []
for model in models :
    catch = model+"_a_lis"
    val.append(hash_bag[catch])

val

"""<h3>COMMISIVE"""

models  = ["b","d","k","r","s"]
cval = []
for model in models :
    catch = model+"_c_lis"
    cval.append(hash_bag[catch])

cval

"""### DIRECT"""

models  = ["b","d","k","r","s"]
dval = []
for model in models :
    catch = model+"_d_lis"
    dval.append(hash_bag[catch])
dval

"""###  EXPRESSIVE"""

models  = ["b","d","k","r","s"]
elist = []
for model in models :
    catch = model+"_e_lis"
    elist.append(hash_bag[catch])
elist



"""RECALL"""

import matplotlib.pyplot as plt

# Given list and model
l = [['assertive', '0.65', '0.350', '0.00', '21'],
     ['assertive', '0.79', '0.71', '0.75', '21'],
     ['assertive', '1.00', '0.10', '0.17', '21'],
     ['assertive', '0.94', '0.71', '0.81', '21'],
     ['assertive', '0.92', '0.57', '0.71', '21']]
model = ["b","d","k","r","s"]

# Extract index 2 values and convert to float
values = [float(x[2]) for x in l]

# Set x-axis labels as model
x_labels = model

# Create a bar chart and set color based on x-axis
bars = plt.bar(x_labels, values)
for i in range(len(x_labels)):
    if x_labels[i] == "b":
        bars[i].set_color("red")
    elif x_labels[i] == "d":
        bars[i].set_color("blue")
    elif x_labels[i] == "k":
        bars[i].set_color("green")
    elif x_labels[i] == "r":
        bars[i].set_color("purple")
    else:
        bars[i].set_color("orange")

# Set the title and y-axis label
plt.title("Recall for assertive")
plt.ylabel("Values")

# Set the background color to dark and grid color to light
plt.style.use('dark_background')
plt.grid(color='grey', alpha=0.3)

# Display the plot
plt.show()

import matplotlib.pyplot as plt

# Given list and model
l = [['commissive', '0.00', '0.00', '0.00', '49'],
     ['commissive', '0.86', '0.78', '0.82', '49'],
     ['commissive', '0.50', '0.02', '0.04', '49'],
     ['commissive', '0.86', '0.78', '0.82', '49'],
     ['commissive', '0.90', '0.78', '0.84', '49']]
model = ["b","d","k","r","s"]

# Extract index 2 values and convert to float
values = [float(x[2]) for x in l]

# Set x-axis labels as model
x_labels = model

# Create a bar chart and set color based on x-axis
bars = plt.bar(x_labels, values)
for i in range(len(x_labels)):
    if x_labels[i] == "b":
        bars[i].set_color("red")
    elif x_labels[i] == "d":
        bars[i].set_color("blue")
    elif x_labels[i] == "k":
        bars[i].set_color("green")
    elif x_labels[i] == "r":
        bars[i].set_color("purple")
    else:
        bars[i].set_color("orange")

# Set the title and y-axis label
plt.title("Recall for commissive")
plt.ylabel("Values")

# Set the background color to dark and grid color to light
plt.style.use('dark_background')
plt.grid(color='grey', alpha=0.3)

# Display the plot
plt.show()



import matplotlib.pyplot as plt

# Given list and model
l = [['directive', '0.00', '0.00', '0.00', '59'],
     ['directive', '0.96', '0.76', '0.85', '59'],
     ['directive', '1.00', '0.14', '0.24', '59'],
     ['directive', '0.96', '0.76', '0.85', '59'],
     ['directive', '0.96', '0.76', '0.85', '59']]
model = ["b","d","k","r","s"]

# Extract index 2 values and convert to float
values = [float(x[2]) for x in l]

# Set x-axis labels as model
x_labels = model

# Create a bar chart and set color based on x-axis
bars = plt.bar(x_labels, values)
for i in range(len(x_labels)):
    if x_labels[i] == "b":
        bars[i].set_color("red")
    elif x_labels[i] == "d":
        bars[i].set_color("blue")
    elif x_labels[i] == "k":
        bars[i].set_color("green")
    elif x_labels[i] == "r":
        bars[i].set_color("purple")
    else:
        bars[i].set_color("orange")

# Set the title and y-axis label
plt.title("Recall for directive")
plt.ylabel("Values")

# Set the background color to dark and grid color to light
plt.style.use('dark_background')
plt.grid(color='grey', alpha=0.3)

# Display the plot
plt.show()



import matplotlib.pyplot as plt

# Given list and model
l = [['expressive', '0.94', '1.00', '0.97', '1874'],
     ['expressive', '0.99', '1.00', '0.99', '1874'],
     ['expressive', '0.94', '1.00', '0.97', '1874'],
     ['expressive', '0.99', '1.00', '0.99', '1874'],
     ['expressive', '0.98', '1.00', '0.99', '1874']]
model = ["b","d","k","r","s"]

# Extract index 2 values and convert to float
values = [float(x[2]) for x in l]

# Set x-axis labels as model
x_labels = model

# Create a bar chart and set color based on x-axis
bars = plt.bar(x_labels, values)
for i in range(len(x_labels)):
    if x_labels[i] == "b":
        bars[i].set_color("red")
    elif x_labels[i] == "d":
        bars[i].set_color("blue")
    elif x_labels[i] == "k":
        bars[i].set_color("green")
    elif x_labels[i] == "r":
        bars[i].set_color("purple")
    else:
        bars[i].set_color("orange")

# Set the title and y-axis label
plt.title("Recall for expressive")
plt.ylabel("Values")

# Set the background color to dark and grid color to light
plt.style.use('dark_background')
plt.grid(color='grey', alpha=0.3)

# Display the plot
plt.show()

"""PRECISION"""

import matplotlib.pyplot as plt

# Given list and model
l = [['assertive', '0.65', '0.350', '0.00', '21'],
     ['assertive', '0.79', '0.71', '0.75', '21'],
     ['assertive', '1.00', '0.10', '0.17', '21'],
     ['assertive', '0.94', '0.71', '0.81', '21'],
     ['assertive', '0.92', '0.57', '0.71', '21']]
model = ["b","d","k","r","s"]

# Extract index 2 values and convert to float
values = [float(x[1]) for x in l]

# Set x-axis labels as model
x_labels = model

# Create a bar chart and set color based on x-axis
bars = plt.bar(x_labels, values)
for i in range(len(x_labels)):
    if x_labels[i] == "b":
        bars[i].set_color("red")
    elif x_labels[i] == "d":
        bars[i].set_color("blue")
    elif x_labels[i] == "k":
        bars[i].set_color("green")
    elif x_labels[i] == "r":
        bars[i].set_color("purple")
    else:
        bars[i].set_color("orange")

# Set the title and y-axis label
plt.title("Recall for assertive")
plt.ylabel("Values")

# Set the background color to dark and grid color to light
plt.style.use('dark_background')
plt.grid(color='grey', alpha=0.3)

# Display the plot
plt.show()

