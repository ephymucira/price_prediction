import streamlit as st

from joblib import load                                                               
import numpy as np

loaded_data = load('saved_steps.joblib')
regressor = loaded_data["model"]
le_country = loaded_data["le_country"]
le_education = loaded_data["le_education"]


def show_predict_page():
    st.title("Software Developer Salary")
    st.write("""###Enter the required  information to predict the salary""")

    countries = (
    "United States",
    "India",
    "United Kingdom",
    "Germany",
    "Canada",
    "France",
    "Brazil",
    "Australia",
    "Spain",
    "Poland",
    "Russian Federation",
    "Netherlands",
    "Sweden",
    "Italy",
    "Israel",
    "Switzerland",
    "Turkey",
    "Ukraine",
    "Belgium"
    )
    
    education = (
        "Bachelors degree",
        "Masters degree",
        "Post grad",
        "Less than a Bachelors"
    )
    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level",education)

    experience  = st.slider("Years of Experience",0,50,3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education,experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")


