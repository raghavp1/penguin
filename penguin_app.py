
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):

  if model == "SVC":
    species = svc_model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  if model == "Logistic Regression":
    species = log_reg.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  if model == "Random Forest Classifier":
    species = rf_clf.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])

  if species[0] == 0:
    return "Adelie"
  elif species[0] == 1:
    return "Chinstrap"
  else:
    return "Gentoo"



# Add title widget
st.title("Penguin Species Prediction")  

# Add 4 sliders and store the value returned by them in 4 separate variables.
bill_length_mm = st.sidebar.slider("bill_length_mm", 32.1, 59.6)
bill_depth_mm = st.sidebar.slider("bill_depth_mm", 13.1, 21.5)
flipper_length_mm = st.sidebar.slider("flipper_length_mm", 172.0, 231.0)
body_mass_g = st.sidebar.slider("body_mass_g", 2700, 6300)

island_dict = {'Biscoe': 0, 'Dream': 1, 'Torgersen':2}
island = island_dict[st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))]

sex_dict = {"Male":0, "Female":1}
sex = sex_dict[st.sidebar.selectbox("Sex", ("Male", "Female"))]


model = st.sidebar.selectbox("Classifier", ("SVC", "Logistic Regression", "Random Forest Classifier"))

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.sidebar.button("Predict"):
    if model == "SVC":
        score = svc_score
    if model == "Logistic Regression":
        score = log_reg_score
    if model == "Random Forest Classifier":
        score = rf_clf_score


    species_type = prediction(model, island, bill_length_mm, flipper_length_mm, bill_depth_mm, body_mass_g, sex)
    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", score)


    