# 🚀 Quick Start Guide

## Run the App Locally (3 Simple Steps!)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Open Your Browser
The app will automatically open at:
```
http://localhost:8501
```

---

## 🎯 What Can You Do?

### 1. Make Predictions
- Enter student academic details
- Get instant placement prediction
- See confidence score

### 2. View Insights
- Check model performance
- See confusion matrices
- Understand feature importance

### 3. Get Recommendations
- Personalized advice for students
- Key factors affecting placement
- Areas to improve

---

## 📊 Test with Sample Data

### High Performer (Expected: Placed)
- Gender: Male
- SSC %: 85, Board: Central
- HSC %: 88, Board: Central, Stream: Science
- Degree %: 82, Type: Sci&Tech
- Work Experience: Yes
- Entrance Test %: 78
- MBA %: 75
- Specialization: Mkt&Fin

### Average Performer (Expected: Not Placed)
- Gender: Female
- SSC %: 65, Board: Others
- HSC %: 68, Board: Others, Stream: Commerce
- Degree %: 62, Type: Comm&Mgmt
- Work Experience: No
- Entrance Test %: 60
- MBA %: 58
- Specialization: Mkt&HR

---

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn pillow
```

### Issue: "Model file not found"
**Solution:**
```bash
python eda.py  # Run this first to create the model
```

### Issue: Port already in use
**Solution:**
```bash
streamlit run app.py --server.port 8502
```

---

## 📁 Project Files

✅ `app.py` - Streamlit web application  
✅ `eda.py` - Complete ML pipeline  
✅ `placement_prediction_model.pkl` - Trained model  
✅ `label_encoders.pkl` - Encoding mappings  
✅ `requirements.txt` - Dependencies  
✅ `README.md` - Full documentation  
✅ `DEPLOYMENT.md` - Deployment guide  

---

## 🎓 Next Steps

1. ✅ Test the app locally
2. ✅ Try different student profiles
3. ✅ Deploy to Streamlit Cloud (FREE!)
4. ✅ Share with friends and recruiters
5. ✅ Add to your portfolio

---

## 💡 Tips

- The app runs locally on your computer
- No internet needed (except for initial package install)
- Changes to `app.py` auto-reload in browser
- Press `Ctrl+C` in terminal to stop the app

---

**Enjoy your Machine Learning web app! 🎉**
