import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

import os
import sys

# statistical method classifiers
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge

# ML method classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Tools for dimension reduction
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis

# path to the dataset file	
FILE_PATH = 'NSL-KDD-Dataset.csv'

st.title("Network Traffic Data Analysis")
st.write("""
	Apply various statistical and machine learning methods from scikit-learn on a sample of the NSL-KDD 99 dataset of network packets and let them predict whether packets from the test set belong to a normal traffic, or to an attack attempt.
""")

# UI widgets for selecting the dataset size
dataset_size = st.sidebar.slider("Dataset Size", min_value = 1, max_value = 120000, value = 5000)
test_set_ratio = st.sidebar.slider("Test Set Ratio", min_value = 0.1, max_value = 0.9, value = 0.2)

# UI selection for the ML method
selected_method = st.sidebar.selectbox("Evaluation Method", ("Linear Regression", "Lasso", "Bayesian Ridge Regression", "KNN (K-Neighbors)", "SVM (Support Vector)", "Random Forest"), index=5)

# Creating the UI widgets for manipulating algorithm parameters
def show_parameter_ui(method_name):
    params = dict()
    
    # Lasso method
    if (method_name == "Lasso"):
    	alpha = st.sidebar.slider("Alpha", min_value = 0.01, max_value = 1.0, value = 1.0)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
    	params["alpha"] = alpha
    
    # Bayesian Ridge Regression method
    elif (method_name == "Bayesian Ridge Regression"):
    	lambda_init = st.sidebar.slider("Initial Lambda", min_value = 0.01, max_value = 3.0, value = 1.0)
    	alpha_init = st.sidebar.slider("Initial Alpha", min_value = 0.01, max_value = 3.0, value = 1.0)
    	params["lambda_init"] = lambda_init
    	params["alpha_init"] = alpha_init

    # KNN method
    elif (method_name == "KNN (K-Neighbors)"):
        K = st.sidebar.slider("K", min_value = 1, max_value = 15, value = 1)
        params["K"] = K

    # SVM method
    elif (method_name == "SVM (Support Vector)"):
        C = st.sidebar.slider("C", min_value = 0.01, max_value = 10.0)
        params["C"] = C

    # Random Forest method
    elif (method_name == "Random Forest"):
        max_depth = st.sidebar.slider("Max Depth", min_value = 2, max_value = 15)
        n_estimators = st.sidebar.slider("Number of Estimators", min_value = 1, max_value = 100)
        random_seed = st.sidebar.slider("Random Seed", min_value = 1, max_value = 99999)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["random_seed"] = random_seed

    return params

params = show_parameter_ui(selected_method)

# Returning the classifier depending on the chosen ML method
def get_classifier(method_name, params):
    # Linear Regression
    if (method_name == "Linear Regression"):
    	clf = LinearRegression()
    	
    # Lasso
    elif (method_name == "Lasso"):
    	clf = Lasso(alpha = params["alpha"])
    	
    # Bayesian Ridge Regression
    elif (method_name == "Bayesian Ridge Regression"):
    	clf = BayesianRidge(alpha_init = params["alpha_init"], lambda_init = params["lambda_init"]	)
    	
    # KNN method
    elif (method_name == "KNN (K-Neighbors)"):
        clf = KNeighborsClassifier(n_neighbors = params["K"])

    # SVM method
    elif (method_name == "SVM (Support Vector)"):
        clf = SVC(C = params["C"])

    # Random Forest method
    else:
        clf = RandomForestClassifier(max_depth = params["max_depth"], n_estimators = params["n_estimators"], random_state = params["random_seed"])

    return clf

clf = get_classifier(selected_method, params)

# Reading the data
header_names = ['duration', 'protocol_type', 'service', 'flag', 
                'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 
                'num_root', 'num_file_creations', 'num_shells', 
                'num_access_files', 'num_outbound_cmds', 
                'is_host_login', 'is_guest_login', 'count', 
                'srv_count', 'serror_rate', 'srv_serror_rate', 
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate', 
                'dst_host_count', 'dst_host_srv_count', 
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
                'attack_type', 'success_pred']

# Loads n rows of KDD-90 data set
def load_data(nrows):
    full_path = os.path.join(os.path.dirname(sys.path[0]), FILE_PATH)
    data = pd.read_csv(full_path, nrows=nrows, names=header_names)
    return data

data = load_data(dataset_size)

# Displaying the data
st.write("Data Shape: ", data.shape)
st.write("Base Data:")
st.write(data)

# Encoding the non-numerical data, creating a category for each value type
def encode_text_dummy(data, name):
    dummies = pd.get_dummies(data[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        data[dummy_name] = dummies[x]
    data.drop(name, axis=1, inplace=True)

encode_text_dummy(data, 'protocol_type')
encode_text_dummy(data, 'service')
encode_text_dummy(data, 'flag')
encode_text_dummy(data, 'land')
encode_text_dummy(data, 'logged_in')
encode_text_dummy(data, 'is_host_login')
encode_text_dummy(data, 'is_guest_login')

# Splitting labels and features
labels = data["attack_type"]
features = data.drop(['attack_type', 'success_pred'], axis = 1)

le = preprocessing.LabelEncoder()
labels_encoded = le.fit(labels).transform(labels)

# Splitting train/test set
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size = test_set_ratio, random_state = 1234)

# Training
model = clf.fit(X_train, y_train)

# Making predictions on the test set
y_pred = clf.predict(X_test)
y_pred_rounded = [round(num) for num in y_pred]

st.write(f"Classifier: {selected_method}")

# Showing the results
st.write("Test Set Results:")
X_test["test_packet_attack_type"] = le.inverse_transform(y_test)
X_test["prediction"] = le.inverse_transform(y_pred_rounded)
X_test["correct"] = (y_test == y_pred_rounded)
st.write(X_test)

# Calculating accuracy of our predictions
acc = accuracy_score(y_test, y_pred_rounded)
st.write(f"Prediction Accuracy: {acc}")

# Showing prediction graphs
def show_prediction_graph(data, feature):
	predicted = data.copy()
	predicted["test_packet_attack_type"] = predicted["prediction"]
	
	predicted["data_type"] = 'predicted'
	data["data_type"] = 'result'
	
	merged_data = pd.merge(data, predicted)
	
	fig = px.scatter(merged_data, x=feature, y="test_packet_attack_type", color="data_type")
	fig.show()
	st.plotly_chart(fig)
	
st.write("Prediction graph:")

feature_names = [f for f in list(X_test.columns.values) if f not in ['test_packet_attack_type', 'prediction', 'correct']]
show_prediction_on_feature = st.selectbox("Feature", feature_names)

show_prediction_graph(X_test, show_prediction_on_feature)

# Showing the label distribution among testing and training data
display_label_dist = st.checkbox("Show label distribution (attack type)")
if display_label_dist:
    st.markdown("Attack type distribution in the data set:")
    st.bar_chart(data["attack_type"])
	
# Showing the graphs of feature importance in descending order (Random Forest only)
def show_feature_importance(model, data):
	importance_list = pd.DataFrame({'Feature' : X_train.columns, 'Importance' : (model.feature_importances_).astype(float)})
	importance_list = importance_list.sort_values(by='Importance', ascending=False)
	importance_list
	
	fig = px.bar(x=importance_list['Feature'], y=importance_list['Importance'])
	fig.show()
	st.plotly_chart(fig)
	
if (selected_method == "Random Forest"):
	showing_feature_importance = st.checkbox("Show feature importance")
	if showing_feature_importance:
			show_feature_importance(model, data)

# Plotting each of the features on the attack type
def show_features_graphs(data):
	data_labels = data["attack_type"]
	data_features = data.drop(['attack_type', 'success_pred'], axis = 1)
		
	for feature in data_features:
		display_feature = st.checkbox(feature)
		if (display_feature):
			fig = px.scatter(data, x=feature, y="attack_type", color="attack_type")
			fig.show()
			st.plotly_chart(fig)

display_individual_features = st.checkbox("Show individual feature influence")
if display_individual_features:
	show_features_graphs(data)
