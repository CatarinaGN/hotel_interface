import streamlit as st

st.title("Home Page")

# Center the logo
st.image("assets/logo(1).png", use_container_width=True)

# Explanation text below the logo
st.write("""
    
This interface is designed for **Hotel H's Marketing Team** to explore the hotels's customer segmentation data through
         visualization and analysis tools. The team can **compare clusters** by visualizing 
         the distribution of both numeric and categorical features. The available visualizations include histograms, 
         violin plots, radar plots, and bar plots, which can be customized to display up to four different graphs for 
         metric features, allowing for an in-depth comparison across clusters. For categorical features, bar plots are 
         available to show the distribution of values across clusters. Additionally, advanced dimensionality reduction 
         techniques like t-SNE enable users to view the data in both 2D and 3D spaces, making it easier to understand 
         cluster relationships and patterns.

It is also possible to **predict the segment of new customers**. Predictions are made based on the relationship of the 
         input record to the existing clusters. If the record aligns closely with a predefined cluster, it is assigned 
         to the cluster with the smallest distance using a **distance-based metric** (squared distances between the input record
          and all cluster centroids). However, if the record is identified as an **outlier**, it is reclassified using a 
         **Decision Tree Classifier**.

The decision tree model is trained on the cluster data and provides predictions without requiring feature scaling, 
         making it an effective tool for handling outliers. The model also offers an accuracy score, allowing users to evaluate
         its performance on customer classification. This ensures that all records, even those outside the typical cluster
         definitions, are assigned to a relevant cluster.   

In the last capability, you can make your own **suggestions** about the visualizations we have right now, or you can save the 
        ideas you had during exploration. These are saved in a file so the team manager can organize the topics for discussion
        in upcoming meetings.      
""")
## 1st paragraph: done
## 3rd paragraph: done
## 4th paragraph: do we leave this? yes we have :p


