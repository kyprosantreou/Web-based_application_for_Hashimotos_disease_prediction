import pandas as pd
import streamlit as st
from data_plot import *
from random_forest import *
from sklearn.preprocessing import StandardScaler

# Function to load the dataset
def load_dataframe():
    # Load data from CSV file
    data = pd.read_csv("hashimoto.csv")
    
    return data

def main():
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Information", "Data Frame","Random Forest Algorithm", "Hashimoto Prediction"])

    # Load data
    data = load_dataframe()
    data_encoded = pd.get_dummies(data, columns=['sex'])

    scaler = StandardScaler()


    with tab1:
        # Information Tab
        # Display general information about Hashimoto's disease
        st.title("About This App")
        st.write("""
        This web application helps in predicting Hashimoto's disease based on certain parameters.
        
        ### How to Use:
        1. **Data Frame Tab**: View the dataset used for training and testing.
        2. **Random Forest Algorithm Tab**: Visualize the dataset and see the accuracy of the Random Forest classifier.
        3. **Hashimoto Prediction Tab**: Input your information and get a prediction about whether you have Hashimoto's disease or not.
        
        ### What is Hashimoto’s disease?
        Hashimoto’s disease is an autoimmune disorder that can cause hypothyroidism, or underactive thyroid. Rarely, the disease can cause hyperthyroidism, or overactive thyroid.

        The thyroid is a small, butterfly-shaped gland in the front of your neck. In people with Hashimoto’s disease the immune system makes antibodies that attack the thyroid gland
        large numbers of white blood cells, which are part of the immune system, build up in the thyroid
        the thyroid becomes damaged and can’t make enough thyroid hormones
        Thyroid hormones control how your body uses energy, so they affect nearly every organ in your body—even the way your heart beats.        
        
        ### How common is Hashimoto’s disease?
        The number of people who have Hashimoto’s disease in the United States is unknown. However, the disease is the most common cause of hypothyroidism, which affects about 5 in 100 Americans.
                 
        ### What are the symptoms of Hashimoto’s disease?
        Many people with Hashimoto’s disease have no symptoms at first. As the disease progresses, you may have one or more of the symptoms of hypothyroidism.

        Some common symptoms of hypothyroidism include:

        1. Fatigue
        2. Weight gain
        3. Trouble tolerating cold
        4. Joint and muscle pain
        5. Constipation
        6. Dry skin or dry, thinning hair
        7. Heavy or irregular menstrual periods or fertility problems
        8. Slowed heart rate
                 
        Hashimoto’s disease causes your thyroid to become damaged. Most people with Hashimoto’s disease develop hypothyroidism. Rarely, early in the course of the disease, thyroid damage may lead to the release of too much thyroid hormone into your blood, causing symptoms of hyperthyroidism
        Your thyroid may get larger and cause the front of the neck to look swollen. The enlarged thyroid, called a goiter, may create a feeling of fullness in your throat, though it is usually not painful. After many years, or even decades, damage to the thyroid may cause the gland to shrink and the goiter to disappear.
        """)

        st.write("""<small>Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK). <a href="https://www.niddk.nih.gov/health-information/endocrine-diseases/hashimotos-disease">link</a></small> """, unsafe_allow_html=True)

    with tab2:
        # Data Frame Tab
        # Display the dataset used for training and testing
        st.header("Hashimoto Data Frame")
        st.dataframe(data, width=800, height=600)
        plot_data(data_encoded)
    
    with tab3:
        # Random Forest Algorithm Tab
        model, scaler = RandomForest_train(data_encoded)

    with tab4:
        # Hashimoto Prediction Tab
        try:
            # Input fields for user to provide information
            sex  = st.selectbox("Sex", ["Male", "Female"])
            age = st.number_input("How old Are you?", 0, 100, 0)
            family_history  = st.selectbox("Do you have family history", ["No", "Yes"])
            autoimmune  = st.selectbox("Do you suffer from another autoimmune disease?", ["No", "Yes"])
            pregnancy = st.selectbox("Are you pregnant?", ["No", "Yes"])
            tsh = st.number_input("TSH", step=0.01,format="%.3f")
            ft4 = st.number_input("FT4",step=0.01,format="%.3f")
            antibodies_test = st.selectbox("Did you do antibodies test?", ["No", "Yes"])

            new_data = []
            
            if sex == "Male":
                sex = 1
                sex_f = 0
            else:
                sex = 0
                sex_f = 1
            
            if family_history == "Yes":
                family_history = 1
            else:
                family_history = 0
            
            if autoimmune == "Yes":
                autoimmune = 1
            else:
                autoimmune = 0
            
            if pregnancy == "Yes":
                pregnancy = 1
            else:
                pregnancy = 0
            
            if antibodies_test == "Yes":
                tgab = st.number_input("TGAB", step=0.01,format="%.3f")
                tpoab = st.number_input("TPOAB", step=0.01,format="%.3f")
                new_data = [[age, family_history, autoimmune, pregnancy, tsh, ft4, tgab, tpoab,sex_f,sex]]
            else:
                tgab = None
                tpoab = None
                new_data = [[age, family_history, autoimmune, pregnancy, tsh, ft4, tgab, tpoab,sex_f,sex]]
            
            # Make prediction when the submit button is clicked
            if st.button("Submit"):
                if age == 0 or tsh == 0.000 or ft4 == 0.000:
                    st.write("Complete the fields.")
                else:
                    Prediction(model,scaler,new_data)

        except ValueError as ve:
            st.error(f"Value Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
if __name__ == "__main__":
    main()