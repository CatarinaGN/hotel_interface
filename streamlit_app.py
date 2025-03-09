import streamlit as st

#import streamlit_authenticator as stauth

# ---- PAGES SETUP -----
main_page = st.Page(
    page="views/main.py", ## visto e comentado, esclarecer comentários
    title="Home Page",
    icon="🏠",
    default=True
)

cluster_analysis_page = st.Page(
    page="views/cluster_analysis.py", ## mudar
    title="Clusters Analysis",
    icon="📊",
)

cluster_prediction_page = st.Page(
    page="views/cluster_prediction.py", ## mudar
    title="Cluster Prediction",
    icon="🤖",
)

suggestions_page = st.Page(
    page="views/suggestions.py", ## done, mas podemos adicionar o nome final dos clusters em vez de cluster 1,2... 
    title="Your Suggestions",
    icon="📋",
)

# --- Navigation Setup ----

pg = st.navigation(
    {
        "Home": [main_page],
        "Analysis": [cluster_analysis_page, cluster_prediction_page, suggestions_page],
    }
)
#-- on all pages ----
st.logo("assets/logo.png", size="large")
st.sidebar.text("Developed for the Marketing team of Hotel H to take full advantage of the clustering analysis insights.")

pg.run()

## cd C:\Users\marga\OneDrive - Nova SBE\Documents\NOVA IMS\2nd Semester\Business Cases\Cases\Case 1\Github\Case-1\interface\Case_1_interface
## streamlit run streamlit_app.py