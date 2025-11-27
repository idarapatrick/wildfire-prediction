import streamlit as st
import requests
from PIL import Image
import io
import time

# Configuration
API_URL = "https://wildfire-prediction.up.railway.app"


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
    
    # Add helper link for finding wildfire images
    st.info("**Need test images?** Visit [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/map) to find active wildfire satellite imagery worldwide")
    
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Send to API with proper file format
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
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
                
                # Original dataset statistics (from Kaggle - hardcoded to avoid 2GB+ download)
                st.subheader("Original Training Dataset")
                st.write("The model was trained on the Kaggle Wildfire Prediction Dataset:")
                
                # Actual counts from Kaggle dataset (30,250 training images total)
                original_wildfire = 15125
                original_nowildfire = 15125
                original_total = original_wildfire + original_nowildfire
                
                st.bar_chart({"Wildfire": original_wildfire, "No Wildfire": original_nowildfire})
                st.info(f"Original dataset: **{original_total:,}** images with balanced 50/50 class distribution")
                st.caption("Note: Original dataset (~2GB) not included in deployment. Model was pre-trained on this full dataset.")
                
                st.divider()
                
                # Current local subset
                st.subheader("Current Local Subset (For Pipeline Testing)")
                st.write("Smaller representative subset available for API testing and retraining demonstrations:")
                st.bar_chart({"Wildfire": data['wildfire'], "No Wildfire": data['nowildfire']})
                
                total = data['total']
                wild_pct = round((data['wildfire']/total)*100, 1) if total > 0 else 0
                st.info(f"Local subset: **{total}** images ({wild_pct}% wildfire)")
                st.caption("This subset is used for demonstration purposes and incremental retraining via the API.")

                st.divider()

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
    
    # Add helper information
    st.info("**Find real wildfire images:** Use [NASA FIRMS Fire Map](https://firms.modaps.eosdis.nasa.gov/map) to capture active wildfire satellite imagery for training!")
    st.warning("**Smart Retraining:** The model only updates if the new version performs better. Your original trained model is automatically backed up and retained if performance decreases.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload New Training Data")
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
        st.subheader("Trigger Retraining")
        st.write("This will use all available data (old + new) to update the model.")
        
        # Check current training status
        try:
            status_res = requests.get(f"{API_URL}/training_status")
            if status_res.status_code == 200:
                status = status_res.json()
                
                # Show last training result if available
                if status.get("last_training_time"):
                    st.divider()
                    st.caption(f"**Last Training:** {status['last_training_time']}")
                    
                    if status.get("last_training_result") == "success":
                        st.success(f"✅ {status.get('last_training_message', 'Training completed successfully')}")
                    elif status.get("last_training_result") == "error":
                        st.error(f"❌ {status.get('last_training_message', 'Training failed')}")
                
                # Show current status
                if status.get("is_training"):
                    st.warning("⏳ Training in progress... Please wait.")
                    st.info("Refresh this page in a few minutes to see the results.")
                    if st.button("🔄 Refresh Status"):
                        st.rerun()
                else:
                    if st.button("Start Retraining Process"):
                        try:
                            res = requests.post(f"{API_URL}/retrain")
                            response_data = res.json()
                            if "error" in response_data:
                                st.error(response_data["error"])
                            else:
                                st.info(response_data['message'])
                                st.info("⏳ Training started! Click 'Refresh Status' button after a few minutes to see results.")
                                time.sleep(2)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to trigger training: {e}")
        except Exception as e:
            st.error(f"Could not fetch training status: {e}")