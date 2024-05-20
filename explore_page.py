import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]]= categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'other'
    return categorical_map

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def clean_education(x):
    # Convert to lowercase
    x = x.replace("Master's degree (MA, MS, M.Eng., MBA, etc.)", "Masters degree")
    x = x.replace("Bachelor's degree (BA, BS, B.Eng., etc.)", "Bachelors degree")
    x= str(x)
    
    if "Bachelors degree" in x:
        return "Bachelors degree"
    elif "Masters degree" in x:
        return "Masters degree"
    elif "Professional degree (JD, MD, etc.)" in x or 'Other doctoral degree (Ph.D, Ed.D., etc.)' in x:
        return "Post grad"
    else:
        return "Less than a Bachelors"
    
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/HP 840 G3/Downloads/archive (3)/Copy of survey_results_public.csv')
    df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedComp"]]
    df = df.rename({"ConvertedComp":"Salary"}, axis=1)
    df = df[df["Salary"].notnull()]
    df = df.dropna()
    df = df[df["Employment"]== "Employed full-time"]
    df = df.drop("Employment", axis = 1)
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)
    df= df[df['Salary'] <= 250000]
    df= df[df['Salary'] >= 10000]
    df= df[df['Country'] != 'other']
    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel'].apply(clean_education)
    return df

df = load_data() 


def show_explore_page():
    st.title("Explore Sofware Engineers Salary")

    st.write(
        """
     ### Stack Overflow developer survey 
     """
    )

    data = df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)

    st.write("""#### Number of data from different countries""")
    st.pyplot(fig1)

    st.write(
        """
     ### Mean Salary Based on Country
     """
    )
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
     ### Mean Salary Based on Experience
     """
    )
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

