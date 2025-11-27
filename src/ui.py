import streamlit as st
import requests
from PIL import Image
import io

# Configuration
API_URL = "https://wildfire-prediction.onrender.com/"


st.set_page_config(page_title="Wildfire Detection System", layout="wide")

st.title("Satellite Wildfire Detection System")

# Sidebar for System Status
st.sidebar.header("System Status")
try:
    status_response = requests.get(f"{API_URL}/status")
    if status_response.status_code == 200:
        status_data = status_response.json()
        st.sidebar.success(f"System: {status_data['status'].upper()}")
        st.sidebar.info(f"Uptime: {status_data['uptime']}")
    else:
        st.sidebar.error("API is Offline")
except:
    st.sidebar.error("Could not connect to API")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Retraining Pipeline"])


with tab1:
    st.header("Real-time Prediction")
    uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Send to API
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        pred = result['prediction']
                        conf = result['confidence']
                        
                        if pred == "Wildfire Detected":
                            st.error(f"{pred} ({conf}%)")
                        else:
                            st.success(f"{pred} ({conf}%)")
                    else:
                        st.error("Error from API")
                except Exception as e:
                    st.error(f"Connection Error: {e}")


with tab2:
    st.header("Dataset Visualization & Interpretation")
    st.write("Interpretations of features in the current training dataset.")
    
    if st.button("Refresh Statistics"):
        try:
            response = requests.get(f"{API_URL}/data_stats")
            if response.status_code == 200:
                data = response.json()
                
                # Visualization 1: Class Balance
                st.subheader("1. Class Distribution Balance")
                st.write("This chart shows if our model is biased towards one category.")
                st.bar_chart({"Wildfire": data['wildfire'], "No Wildfire": data['nowildfire']})
                
                # Interpretation Text
                total = data['total']
                wild_pct = round((data['wildfire']/total)*100, 1) if total > 0 else 0
                st.info(f"Interpretation: The dataset contains {total} images. {wild_pct}% are wildfire examples.")

                # Visualization 2: Metric Explanation (Static for this assignment)
                st.subheader("2. Input Feature: RGB Histogram Theory")
                st.write("Wildfire images typically possess higher Red channel intensity compared to Green/Blue.")
                st.progress(70) # Visual placeholder for Red channel importance
                st.caption("Red Channel Importance (Estimated)")

            else:
                st.warning("Could not fetch stats.")
        except:
            st.error("API Connection Failed")


with tab3:
    st.header("Model Retraining Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload New Training Data")
        label_option = st.selectbox("Select Class Label", ["wildfire", "nowildfire"])
        bulk_files = st.file_uploader("Upload multiple images", accept_multiple_files=True)
        
        if st.button("Upload to Server"):
            if bulk_files:
                files_payload = [('files', (file.name, file.getvalue(), file.type)) for file in bulk_files]
                try:
                    res = requests.post(
                        f"{API_URL}/upload_data", 
                        params={"label": label_option},
                        files=files_payload
                    )
                    st.success(res.json()['message'])
                except Exception as e:
                    st.error(f"Upload failed: {e}")
            else:
                st.warning("Please choose files first.")

    with col2:
        st.subheader("2. Trigger Retraining")
        st.write("This will use all available data (old + new) to update the model.")
        
        if st.button("Start Retraining Process"):
            try:
                res = requests.post(f"{API_URL}/retrain")
                st.info(res.json()['message'])
                st.caption("Check your terminal/console for training progress logs.")
            except:
                st.error("Failed to trigger training.")