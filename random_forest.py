from main import *
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Function to train Random Forest model
def RandomForest_train(data_encoded):
    # Convert columns to numeric
    numeric_cols = ['age', 'family history', 'other autoimmune', 'pregnancy', 'tsh', 'ft4', 'tgab', 'tpoab', 'Hashimoto', 'sex_f']
    # data_encoded[numeric_cols] = data_encoded[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Splitting the data
    X = data_encoded.drop('Hashimoto', axis=1)
    y = data_encoded['Hashimoto']


    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    st.write("Xtrain:", len(X_train))
    st.write("Xtest:",len(X_test))
    print("Train ", X_train,"\n")
    print("Test ", X_test,"\n")

    # Training the random forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    st.write("Accuracy:",model.score(X_test, y_test))

    y_pred = model.predict(X_test)

    # Calculate and plotting confusion matrix 
    cm = confusion_matrix(y_test, y_pred)

    print(classification_report(y_test,y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',xticklabels=['Healthy', 'Hashimoto'], yticklabels=['Healthy', 'Hashimoto'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

    print(classification_report(y_test,y_pred))

    return model, scaler

# Function to make predictions
def Prediction(model,scaler,new_data):
    # Predict whether the person has Hashimoto's disease
    prediction = model.predict(new_data)

    new_data_scaled = scaler.transform(new_data)

    # Predict whether the person has Hashimoto's disease
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)
    
    # Probability of Hashimoto's disease
    hashimoto_prob = prediction_proba[0][1]  

    # Display probability as a percentage
    hashimoto_prob_percentage = hashimoto_prob * 100

    st.write(f"Probability of having Hashimoto's disease: {hashimoto_prob_percentage:.2f}%")

    if hashimoto_prob >= 0.5:
        st.write("The person may have Hashimoto's disease.")
    else:
        st.write("The person most likely does not have the disease.")

if __name__ == "__main__":
    main()