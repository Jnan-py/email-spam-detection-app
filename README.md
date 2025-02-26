# Email Spam Detection App

The **Email Spam Detection App** is a Streamlit-based web application that classifies email content as spam or ham (not spam). The app uses a Naive Bayes classifier along with a CountVectorizer to process and predict whether an email is spam. In addition, it offers a range of interactive data visualizations including distribution plots, histograms, and word clouds for both spam and ham messages.

## Features

- **Spam Prediction:**
  - Input your email content and receive a prediction on whether the email is spam or not.
- **Model Training and Persistence:**
  - If a trained model and vectorizer are not found locally, the app automatically trains a new model on the provided dataset and saves them for future use.
- **Data Visualizations:**
  - **Dataset Preview:** View a preview of the dataset.
  - **Category Distribution:** Pie charts to visualize the proportion of spam vs ham emails.
  - **Message Length Distribution:** Histogram showing the distribution of message lengths.
  - **Word Clouds:** Visualize the most common words in spam and ham messages.
  - **Bar Plot:** Bar chart to display the count of spam and ham emails.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/email-spam-detection-app.git
   cd email-spam-detection-app
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Dataset Setup:**
   - Ensure the dataset file is located at `Dataset/spam.csv`.
   - The CSV should have at least two columns: one for the category (with values such as "spam" and "ham") and one for the email message.

## Usage

1. **Run the Application:**

   ```bash
   streamlit run main.py
   ```

2. **Spam Prediction:**

   - Navigate to the **Spam Prediction** tab.
   - Enter your email content in the text area and click the **Predict** button.
   - The app will display whether the email is classified as spam or not.

3. **Data Visualization:**
   - Navigate to the **Data Visualization** tab.
   - Explore various visualizations such as:
     - Dataset preview with a selectable number of rows.
     - Spam vs Ham distribution (pie chart).
     - Message length distribution (histogram).
     - Word clouds for both spam and ham messages.
     - Bar plot of spam vs ham counts.

## Project Structure

```
email-spam-detection-app/
│
├── main.py                  # Main Streamlit application file
├── Dataset/
│   └── spam.csv            # Dataset containing email messages and their labels
├── spam_classifier.pkl     # Saved Naive Bayes model (generated on first run)
├── vectorizer.pkl          # Saved CountVectorizer (generated on first run)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web interface.
- **Pandas & NumPy:** For data manipulation and numerical computations.
- **Scikit-Learn:** For training and evaluating the Naive Bayes spam classifier.
- **Plotly & Matplotlib:** For interactive visualizations.
- **WordCloud:** For generating word clouds to visualize text data.

---

Save these files in your project directory. To run the app, activate your virtual environment (if using one) and run:

```bash
streamlit run main.py
```

Feel free to modify the documentation as needed. Enjoy your Email Spam Detection App!
