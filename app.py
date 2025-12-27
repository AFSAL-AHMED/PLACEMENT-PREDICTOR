import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Placement Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .placed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .not-placed {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('placement_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please run eda.py first.")
        return None

model = load_model()

st.markdown("<h1>🎓 Student Placement Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #6b7280;'>Predict placement chances using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("## 📊 About")
    st.markdown("""
    <div class='info-box'>
    This system predicts whether an MBA student will be placed or not based on their academic performance and background.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 Model Info")
    st.info("""
    - **Algorithm**: Random Forest
    - **Accuracy**: ~86%
    - **Features**: 12
    - **Training Samples**: 172
    """)
    
    st.markdown("### 📈 Features")
    st.success("""
    ✅ Real-time predictions  
    ✅ Confidence scores  
    ✅ User-friendly interface  
    ✅ Detailed insights  
    """)

tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Model Insights", "ℹ️ How to Use"])

with tab1:
    st.markdown("## 📝 Enter Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 Academic Information")
        
        st.markdown("**Secondary School (SSC)**")
        ssc_p = st.slider("SSC Percentage", 40.0, 100.0, 70.0, 0.5)
        ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
        
        st.markdown("**Higher Secondary (HSC)**")
        hsc_p = st.slider("HSC Percentage", 40.0, 100.0, 70.0, 0.5)
        hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
        hsc_s = st.selectbox("HSC Stream", ["Arts", "Commerce", "Science"])
        
        st.markdown("**Undergraduate Degree**")
        degree_p = st.slider("Degree Percentage", 40.0, 100.0, 70.0, 0.5)
        degree_t = st.selectbox("Degree Type", ["Comm&Mgmt", "Others", "Sci&Tech"])
    
    with col2:
        st.markdown("### 💼 MBA & Experience")
        
        workex = st.radio("Work Experience", ["No", "Yes"])
        
        st.markdown("**MBA Information**")
        etest_p = st.slider("Entrance Test Percentage", 40.0, 100.0, 70.0, 0.5)
        mba_p = st.slider("MBA Percentage", 40.0, 100.0, 70.0, 0.5)
        specialisation = st.selectbox("MBA Specialization", ["Mkt&Fin", "Mkt&HR"])
        
        st.markdown("**Personal Information**")
        gender = st.radio("Gender", ["Male", "Female"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 PREDICT PLACEMENT", use_container_width=True)
    
    if predict_button:
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
        else:
            gender_encoded = 1 if gender == "Male" else 0
            ssc_b_encoded = 0 if ssc_b == "Central" else 1
            hsc_b_encoded = 0 if hsc_b == "Central" else 1
            hsc_s_encoded = {"Arts": 0, "Commerce": 1, "Science": 2}[hsc_s]
            degree_t_encoded = {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2}[degree_t]
            workex_encoded = 1 if workex == "Yes" else 0
            specialisation_encoded = 0 if specialisation == "Mkt&Fin" else 1
            
            input_data = np.array([[
                gender_encoded, ssc_p, ssc_b_encoded, hsc_p, hsc_b_encoded, hsc_s_encoded,
                degree_p, degree_t_encoded, workex_encoded, etest_p,
                specialisation_encoded, mba_p
            ]])
            
            with st.spinner("Analyzing student profile..."):
                prediction = model.predict(input_data)[0]
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(input_data)[0]
                    confidence = probability[prediction] * 100
                else:
                    confidence = None
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class='prediction-box placed'>
                        ✅ STUDENT WILL BE PLACED
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                    <div class='prediction-box not-placed'>
                        ❌ STUDENT MAY NOT BE PLACED
                    </div>
                """, unsafe_allow_html=True)
            
            if confidence:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Confidence Level", f"{confidence:.2f}%")
                    st.progress(confidence / 100)
            
            st.markdown("---")
            st.markdown("## 💡 Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📚 Academic Profile")
                avg_academic = (ssc_p + hsc_p + degree_p + mba_p) / 4
                st.metric("Average Percentage", f"{avg_academic:.2f}%")
                if avg_academic >= 75:
                    st.success("✅ Strong academic record")
                elif avg_academic >= 60:
                    st.warning("⚠️ Average performance")
                else:
                    st.error("❌ Below average")
            
            with col2:
                st.markdown("### 💼 Experience")
                if workex_encoded == 1:
                    st.success("✅ Has work experience")
                    st.info("Work experience improves placement chances!")
                else:
                    st.warning("⚠️ No work experience")
                    st.info("Consider internships")
            
            with col3:
                st.markdown("### 🎯 MBA Performance")
                st.metric("MBA Percentage", f"{mba_p:.2f}%")
                if mba_p >= 75:
                    st.success("✅ Excellent")
                elif mba_p >= 60:
                    st.warning("⚠️ Good")
                else:
                    st.error("❌ Needs improvement")
            
            st.markdown("---")
            st.markdown("## 📋 Recommendations")
            
            recommendations = []
            
            if mba_p < 70:
                recommendations.append("📈 Focus on improving MBA percentage")
            if workex_encoded == 0:
                recommendations.append("💼 Consider work experience through internships")
            if etest_p < 70:
                recommendations.append("📝 Work on entrance test scores")
            if avg_academic < 70:
                recommendations.append("📚 Overall academic performance needs improvement")
            
            if prediction == 1 and not recommendations:
                st.success("🎉 Excellent profile! Keep it up!")
            elif recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.info("💪 Keep working on your profile!")

with tab2:
    st.markdown("## 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Accuracy Metrics")
        st.markdown("""
        <div class='info-box'>
        <h4>Random Forest Classifier</h4>
        <ul>
        <li><b>Test Accuracy:</b> ~86%</li>
        <li><b>Precision:</b> ~93%</li>
        <li><b>Recall:</b> ~90%</li>
        <li><b>F1-Score:</b> ~92%</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Feature Importance")
        st.markdown("""
        <div class='info-box'>
        <b>Top 5 Important Features:</b>
        <ol>
        <li>MBA Percentage (25%)</li>
        <li>Degree Percentage (19%)</li>
        <li>Entrance Test (17%)</li>
        <li>SSC Percentage (12%)</li>
        <li>Work Experience (10%)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔍 What This Means")
        st.success("""
        **High Precision (93%)**  
        When model predicts "Placed", it's correct 93% of the time.
        """)
        
        st.info("""
        **High Recall (90%)**  
        Catches 90% of all students who will get placed.
        """)
        
        st.warning("""
        **MBA % is Most Important**  
        MBA performance has biggest impact on placement!
        """)
    
    try:
        st.markdown("### 📊 Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            confusion_img = Image.open('confusion_matrices.png')
            st.image(confusion_img, caption="Confusion Matrices", use_container_width=True)
        with col2:
            comparison_img = Image.open('model_comparison.png')
            st.image(comparison_img, caption="Performance Comparison", use_container_width=True)
    except:
        st.info("Visualization images not found. Run eda.py to generate them.")

with tab3:
    st.markdown("## ℹ️ How to Use")
    
    st.markdown("""
    ### 📝 Steps
    
    1. Go to **'Make Prediction'** tab
    2. Enter all student details (percentages, board, stream, etc.)
    3. Click **'Predict Placement'**
    4. View prediction result and confidence
    5. Check insights and recommendations
    
    ### 📊 Understanding Results
    
    - **Placed**: High probability based on profile
    - **Not Placed**: May face challenges, focus on improvements
    - **Confidence**: How certain the model is (higher = better)
    
    ### 🎯 Key Factors
    
    1. **MBA Percentage** - Most important (25% weight)
    2. **Degree Percentage** - Second (19% weight)
    3. **Entrance Test** - Significant (17% weight)
    4. **Work Experience** - Can make big difference!
    5. **Academic Scores** - Consistent performance matters
    
    ### 💡 Tips
    
    ✅ Maintain high MBA percentage (>70%)  
    ✅ Get work experience  
    ✅ Score well in entrance tests  
    ✅ Keep consistent academic performance  
    """)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p><b>Student Placement Prediction System</b></p>
        <p>Built with Python and Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
