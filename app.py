from predict_page import *
from explore_page import * 


page = st.sidebar.selectbox("Explore or predict",("Predict","Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()


