
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Skincare Influencer Dashboard", layout="wide")
st.title("Skincare Influencer Dashboard")

def load_data():
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("synthetic_skincare_influencer_survey.csv")

df = load_data()
st.sidebar.markdown("Download Example CSV [here](synthetic_skincare_influencer_survey.csv)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Visualization", "Classification", "Clustering", "Association Rules", "Regression"
])

with tab1:
    st.header("Data Visualization")
    st.dataframe(df.head())
    st.markdown("### Demographics Breakdown")
    st.bar_chart(df['Gender'].value_counts())
    st.bar_chart(df['Income'].value_counts())
    st.bar_chart(df['MainInfluencerTier'].value_counts())
    st.markdown("### Spend by Platform Used")
    st.write(df.groupby('MostTrustedPlatform')['MonthlySpend'].value_counts().unstack(fill_value=0))
    st.markdown("### Willingness to Try by Tier")
    st.write(df.groupby('MainInfluencerTier')['WillingnessToTry'].value_counts().unstack(fill_value=0))
    st.markdown("### Correlation Matrix (Encoded)")
    enc = LabelEncoder()
    enc_df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            enc_df[col] = enc.fit_transform(df[col])
    st.write(enc_df.corr())
    st.markdown("### Age Distribution")
    st.histogram(df['Age'])

with tab2:
    st.header("Classification")
    st.markdown("Predict 'PurchasedFromInfluencer' using KNN, DT, RF, GBRT")
    X = enc_df.drop(['PurchasedFromInfluencer','WouldRecommend'], axis=1)
    y = (df['PurchasedFromInfluencer']=='Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    models = {
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred)
        })
    st.dataframe(pd.DataFrame(results))
    selected = st.selectbox("Show confusion matrix for model:", list(models.keys()))
    pred = models[selected].predict(X_test)
    cm = confusion_matrix(y_test, pred)
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(cm, index=["No", "Yes"], columns=["Pred_No", "Pred_Yes"]))
    st.markdown("### ROC Curve")
    plt.figure()
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    st.pyplot(plt.gcf())
    st.markdown("### Predict on New Data")
    upload = st.file_uploader("Upload data for prediction (exclude PurchasedFromInfluencer)", type="csv", key="pred")
    if upload:
        new_data = pd.read_csv(upload)
        for col in X.columns:
            if col in new_data and new_data[col].dtype == "object":
                new_data[col] = enc.fit_transform(new_data[col].astype(str))
        pred = models['RandomForest'].predict(new_data[X.columns])
        out = new_data.copy()
        out['Predicted_PurchasedFromInfluencer'] = pred
        st.dataframe(out)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv")

with tab3:
    st.header("Clustering")
    st.markdown("KMeans Clustering on Encoded Features")
    cluster_n = st.slider("Number of Clusters", 2, 10, 4)
    scaler = StandardScaler()
    Xclus = scaler.fit_transform(enc_df)
    inertia = []
    for k in range(2,11):
        km = KMeans(n_clusters=k, random_state=42).fit(Xclus)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertia, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia (Elbow)")
    st.pyplot(fig)
    kmeans = KMeans(n_clusters=cluster_n, random_state=42).fit(Xclus)
    labels = kmeans.labels_
    df['Cluster'] = labels
    st.dataframe(df.groupby('Cluster').agg(lambda x: x.value_counts().index[0]))
    st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv")

with tab4:
    st.header("Association Rules (Apriori)")
    st.markdown("Select columns for association mining (multi-select columns only):")
    apri_cols = st.multiselect("Columns", ['PlatformsUsed','InfluencerContentType','PurchaseInfluencingFactors'])
    min_sup = st.slider("Min Support", 0.01, 0.2, 0.07)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.7)
    if apri_cols:
        # Prepare transaction DataFrame for mlxtend
        all_items = set()
        transactions = []
        for _, row in df[apri_cols].iterrows():
            items = set()
            for val in row:
                items.update([i.strip() for i in val.split(",")])
            transactions.append(list(items))
            all_items.update(items)
        onehot = pd.DataFrame([{item: (item in tran) for item in all_items} for tran in transactions])
        freq = apriori(onehot, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

with tab5:
    st.header("Regression")
    st.markdown("Regression: Predict Age from Encoded Features")
    y = df['Age']
    Xreg = enc_df.drop(['Age','PurchasedFromInfluencer','WouldRecommend','Cluster'], axis=1, errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(Xreg, y, test_size=0.3, random_state=42)
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "DecisionTree": DecisionTreeRegressor()
    }
    insights = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = model.score(X_test, y_test)
        insights.append({"Model": name, "R2": r2, "MAE": np.mean(np.abs(y_test-y_pred))})
    st.dataframe(pd.DataFrame(insights))
    st.markdown("### Age Distribution (Predicted vs Actual for Decision Tree)")
    y_pred = models["DecisionTree"].predict(X_test)
    fig, ax = plt.subplots()
    ax.hist(y_test, alpha=0.5, label='Actual')
    ax.hist(y_pred, alpha=0.5, label='Predicted')
    ax.legend()
    st.pyplot(fig)
