import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Skincare Survey Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("skincare_survey_data.csv")
    return df

df = load_data()

st.title("Skincare Influencer Survey Dashboard")

tabs = st.tabs(["Data Visualization", "Classification", "Download Data"])

with tabs[0]:
    st.header("Data Visualization & Exploration")
    st.write("Sample of dataset:")
    st.dataframe(df.sample(10))

    st.subheader("Demographic Distribution")
    st.bar_chart(df['Age'].value_counts())
    st.bar_chart(df['Gender'].value_counts())

    st.subheader("Income Distribution")
    st.hist(df['Income_Numeric'], bins=30)

    st.subheader("Spend Distribution (Skew & Outliers)")
    st.hist(df['Monthly_Spend'], bins=30)

    st.subheader("Influencer Purchased?")
    st.bar_chart(df['Influencer_Purchased'].value_counts())

    st.subheader("Top Brand Preferences")
    prefs = df['Brand_Preferences'].str.get_dummies(sep=',').sum().sort_values(ascending=False)
    st.bar_chart(prefs)

with tabs[1]:
    st.header("Classification Modeling")
    st.write("Let's predict whether a respondent will purchase via influencer.")

    # Prepare data for classification
    X = df[['Income_Numeric', 'Monthly_Spend']]
    # Add some encoding
    le_age = LabelEncoder()
    X['Age'] = le_age.fit_transform(df['Age'])
    le_gender = LabelEncoder()
    X['Gender'] = le_gender.fit_transform(df['Gender'])
    le_tier = LabelEncoder()
    X['Influencer_Tier'] = le_tier.fit_transform(df['Influencer_Tier'])
    y = df['Influencer_Purchased'].map({'Yes':1, 'No':0})

    X = pd.concat([X, pd.get_dummies(df['Buy_Frequency'])], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append([name, acc, report['1']['precision'], report['1']['recall'], report['1']['f1-score']])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    st.dataframe(results_df)

    st.subheader("Confusion Matrix (select model)")
    sel_model = st.selectbox("Choose a model:", list(models.keys()))
    model = models[sel_model]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write(pd.DataFrame(cm, index=["No", "Yes"], columns=["Pred No", "Pred Yes"]))

    st.subheader("ROC Curve")
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)

with tabs[2]:
    st.header("Download Synthetic Data")
    st.write("Download the full synthetic dataset for your own analysis.")
    st.download_button("Download CSV", df.to_csv(index=False), "skincare_survey_data.csv")
