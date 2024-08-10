from main import *
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

# Function to plot data
def plot_data(data_encoded):
    try:
        # Plot for Age vs TSH
        plt.xlabel('Age')
        plt.ylabel('TSH')
        plt.scatter(data_encoded[data_encoded['Hashimoto'] == 0]['age'], data_encoded[data_encoded['Hashimoto'] == 0]['tsh'], color="blue", marker='*')
        plt.scatter(data_encoded[data_encoded['Hashimoto'] == 1]['age'], data_encoded[data_encoded['Hashimoto'] == 1]['tsh'], color="black", marker='.')
        plt.legend(["Healthy","People with Hashimoto disease"])
        st.pyplot()

        # Plot for Age vs FT4
        plt.xlabel('Age')
        plt.ylabel('FT4')
        plt.scatter(data_encoded[data_encoded['Hashimoto'] == 0]['age'], data_encoded[data_encoded['Hashimoto'] == 0]['ft4'], color="purple", marker='*')
        plt.scatter(data_encoded[data_encoded['Hashimoto'] == 1]['age'], data_encoded[data_encoded['Hashimoto'] == 1]['ft4'], color="black", marker='.')
        plt.legend(["Healthy","People with Hashimoto disease"])
        st.pyplot()
    except Exception as e:
        st.write("Error {e}")

if __name__ == "__main__":
    main()