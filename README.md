# 🎭 SentiMent - Sentiment Analysis Web App

A beautiful and interactive Streamlit web application for real-time sentiment analysis with multiple input options.

## 🌟 Features

- **Beautiful UI**: Gradient backgrounds, smooth animations, and intuitive design
- **Dual Input Options**: 
  - 📚 Select from pre-loaded example sentences
  - ✍️ Type or paste your own text
- **Real-time Analysis**: Instant sentiment classification with confidence scores
- **Three Sentiment Categories**:
  - 😊 **Positive**: Happy, satisfied, enthusiastic
  - 😐 **Neutral**: Factual, objective, informative
  - 😢 **Negative**: Unhappy, angry, disappointed
- **Confidence Visualization**: See how confident the model is in its prediction
- **Smart Model Management**: Automatically trains and saves models for faster loading

## 🚀 Installation

1. **Navigate to the NLP directory**:
   ```bash
   cd "ML intern elysium/NLP"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure you have the dataset** in the correct location:
   ```
   NLP/
   ├── app.py
   ├── requirements.txt
   ├── sentiment-analysis.ipynb
   └── sentimentDataset/
       └── sentiment_analysis.csv
   ```

## 🎬 Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📱 How to Use

### Method 1: Select from Examples
1. Choose **"📚 Select from Examples"** from the radio button
2. Select a sentiment category (Positive, Neutral, or Negative)
3. Pick a pre-loaded sentence
4. Click **"🔮 Analyze Sentiment"** button

### Method 2: Type Your Own
1. Choose **"✍️ Type Your Own"** from the radio button
2. Type or paste your text in the text area
3. Click **"🔮 Analyze Sentiment"** button
4. View instant results with confidence score!

## 🤖 Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF
- **n-grams**: 1-2 (unigrams and bigrams)
- **Max Features**: 5,000
- **Input Length**: Variable
- **Approximate Accuracy**: ~85%

## 📊 Technical Stack

- **Frontend Framework**: Streamlit
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Language**: Python 3.8+

## 🎨 Customization

You can easily customize the app by:

1. **Add more examples** in the `example_sentences` dictionary
2. **Change colors** in the CSS styling section
3. **Modify the model** to use different algorithms
4. **Update the confidence threshold** for different use cases

## ⚠️ Notes

- The app automatically trains the model on first run and saves it for faster loading
- Sarcasm and complex language might not be detected perfectly
- Longer texts generally provide more accurate results
- The model works best with English text

## 🔄 Model Retraining

To retrain the model with updated data:
1. Delete `sentiment_model.pkl` and `vectorizer.pkl` files
2. Run the app again, it will automatically retrain

## 📝 Example Sentences

### Positive 😊
- "I absolutely love this product! It's amazing!"
- "This is the best day of my life!"
- "You're doing a fantastic job!"

### Neutral 😐
- "The weather is cloudy today."
- "I have a meeting at 3 PM."
- "This is a table."

### Negative 😢
- "I hate waiting in long lines."
- "This is absolutely terrible!"
- "I don't like this at all."

## 🐛 Troubleshooting

**Issue**: CSV file not found
- **Solution**: Make sure `sentimentDataset/sentiment_analysis.csv` exists in the NLP folder

**Issue**: Model takes too long to load first time
- **Solution**: This is normal on first run. Subsequent runs will be much faster.

**Issue**: Low accuracy on custom text
- **Solution**: Try using complete sentences rather than fragments

## 📚 Related Files

- `sentiment-analysis.ipynb`: Full model training notebook with detailed analysis
- `sentimentDataset/sentiment_analysis.csv`: Training dataset

## 🤝 Contributing

Feel free to improve the UI, add new features, or enhance the model!

---

**Made with ❤️ using Python & Streamlit**
