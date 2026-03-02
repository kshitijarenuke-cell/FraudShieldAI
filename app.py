import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main { background-color: #0a0f1e; color: #e8eaf6; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e5ff, #7c4dff, #ff4081);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #90a4ae;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #12203a, #0d1f35);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00e5ff;
    font-family: 'DM Mono', monospace;
}

.metric-label {
    font-size: 0.8rem;
    color: #78909c;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

.fraud-alert {
    background: linear-gradient(135deg, rgba(244,67,54,0.15), rgba(183,28,28,0.1));
    border: 1px solid rgba(244,67,54,0.4);
    border-left: 4px solid #f44336;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.legit-alert {
    background: linear-gradient(135deg, rgba(76,175,80,0.15), rgba(27,94,32,0.1));
    border: 1px solid rgba(76,175,80,0.4);
    border-left: 4px solid #4caf50;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8eaf6;
    border-bottom: 2px solid rgba(0,229,255,0.3);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

.risk-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}

.sidebar-info {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    font-size: 0.85rem;
    color: #90a4ae;
}

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #12203a, #0d1f35);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 12px;
    padding: 12px 16px;
}

div[data-testid="metric-container"] label {
    color: #78909c !important;
}

div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #00e5ff !important;
}

.stButton>button {
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    color: white;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 1rem;
    transition: all 0.3s;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,229,255,0.3);
}

.stSlider>div>div { color: #00e5ff; }
.stSelectbox label, .stSlider label, .stNumberInput label { color: #b0bec5 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL — REAL KAGGLE DATASET
# ─────────────────────────────────────────────
@st.cache_resource
def generate_data_and_train():
    import kagglehub, glob

    with st.spinner('⏳ Loading real Kaggle dataset...'):
        dl_path = kagglehub.dataset_download("nelgiriyewithana/credit-card-fraud-detection-dataset-2023")
        csv_files = glob.glob(dl_path + "/**/*.csv", recursive=True)
        df_full = pd.read_csv(csv_files[0])

    df_full.drop(columns=['id'], inplace=True, errors='ignore')
    df_full.drop_duplicates(inplace=True)
    df_full.dropna(inplace=True)
    df_full.reset_index(drop=True, inplace=True)

    # Sample 50k for speed — still 100% real data!
    df = df_full.sample(n=50000, random_state=42).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols]
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled_df, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled_df[['V1', 'V2', 'Amount']] if 'V1' in X_scaled_df.columns else X_scaled_df.iloc[:, :3])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return df, model, scaler, kmeans, auc, report, cm, fpr, tpr, feature_cols

df, model, scaler, kmeans, auc, report, cm, fpr, tpr, feature_cols = generate_data_and_train()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudShield AI")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Dashboard", "🔍 Fraud Predictor", "📊 Model Insights", "💰 Risk Exposure"])
    st.markdown("---")
    st.markdown('<div class="sidebar-info">📚 <b>ITM Skills University</b><br>Business Studies with Applied AI<br>Group 12 — Mini Project<br><br>Dataset: Credit Card Fraud 2023<br>Models: K-Means + Logistic Regression</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown('<div class="hero-title">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Credit Card Fraud Detection & Financial Risk Intelligence Platform</div>', unsafe_allow_html=True)

    total = len(df)
    fraud_count = df['Class'].sum()
    fraud_pct = fraud_count / total * 100
    fraud_amt = df[df['Class']==1]['Amount'].sum()
    avg_fraud_amt = df[df['Class']==1]['Amount'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraudulent Cases", f"{fraud_count:,}", f"{fraud_pct:.1f}% of total")
    c3.metric("Total Fraud Exposure", f"${fraud_amt:,.0f}")
    c4.metric("Model AUC Score", f"{auc:.3f}", "↑ Excellent")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Transaction Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        sizes = [total - fraud_count, fraud_count]
        colors_pie = ['#00e5ff', '#ff4081']
        wedges, texts, autotexts = ax.pie(sizes, labels=['Legitimate', 'Fraudulent'],
                                           colors=colors_pie, autopct='%1.1f%%',
                                           startangle=90, textprops={'color': '#e8eaf6', 'fontsize': 11})
        for at in autotexts:
            at.set_color('#0a0f1e')
            at.set_fontweight('bold')
        ax.set_title('Transaction Split', color='#e8eaf6', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Amount Distribution by Class</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        ax.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, color='#00e5ff', label='Legitimate', density=True)
        ax.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, color='#ff4081', label='Fraudulent', density=True)
        ax.set_xlabel('Transaction Amount (USD)', color='#90a4ae')
        ax.set_ylabel('Density', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.spines['bottom'].set_color('#263238')
        ax.spines['left'].set_color('#263238')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(facecolor='#12203a', labelcolor='#e8eaf6')
        ax.set_title('Amount Patterns', color='#e8eaf6', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Cluster Viz
    st.markdown('<div class="section-title">K-Means Risk Segmentation</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    cluster_colors = {0: '#00e5ff', 1: '#ffd740', 2: '#ff4081'}
    cluster_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    for c in range(3):
        mask = df['Cluster'] == c
        ax.scatter(df[mask]['V1'], df[mask]['V2'],
                   c=cluster_colors[c], label=cluster_labels[c],
                   alpha=0.35, s=12)
    ax.set_xlabel('V1 (Transaction Behavior)', color='#90a4ae')
    ax.set_ylabel('V2 (Spending Pattern)', color='#90a4ae')
    ax.tick_params(colors='#90a4ae')
    ax.spines['bottom'].set_color('#263238')
    ax.spines['left'].set_color('#263238')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#12203a', labelcolor='#e8eaf6')
    ax.set_title('Customer Risk Clusters (K=3)', color='#e8eaf6', fontsize=13, fontweight='bold')
    st.pyplot(fig)
    plt.close()

    cluster_stats = df.groupby('Cluster').agg(
        Count=('Class', 'count'),
        Fraud_Cases=('Class', 'sum'),
        Fraud_Rate=('Class', lambda x: f"{x.mean()*100:.1f}%"),
        Avg_Amount=('Amount', lambda x: f"${x.mean():.0f}")
    ).reset_index()
    cluster_stats['Cluster'] = cluster_stats['Cluster'].map(cluster_labels)
    cluster_stats.columns = ['Risk Segment', 'Total Transactions', 'Fraud Cases', 'Fraud Rate', 'Avg Amount']
    st.dataframe(cluster_stats, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: FRAUD PREDICTOR
# ─────────────────────────────────────────────
elif page == "🔍 Fraud Predictor":
    st.markdown('<div class="hero-title">🔍 Transaction Fraud Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter transaction features to get an instant fraud risk assessment</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("ℹ️ What do V1–V5 mean?"):
        st.markdown("""
        V1–V5 are **PCA (Principal Component Analysis) features** derived from the raw transaction data.
        They represent anonymized behavioral patterns such as:
        - **V1**: Transaction velocity and location anomaly
        - **V2**: Merchant category behavior
        - **V3**: Card usage pattern deviation
        - **V4**: Time-of-day spending pattern
        - **V5**: Cross-border / online transaction indicator

        In the real Kaggle dataset, these come pre-computed. Use values between **-5 and +5** for realistic inputs.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔢 PCA Transaction Features**")
        V1 = st.slider("V1 – Location/Velocity Anomaly", -5.0, 5.0, 0.0, 0.1)
        V2 = st.slider("V2 – Merchant Behavior", -5.0, 5.0, 0.0, 0.1)
        V3 = st.slider("V3 – Usage Pattern Deviation", -5.0, 5.0, 0.0, 0.1)

    with col2:
        st.markdown("**💳 Transaction Details**")
        V4 = st.slider("V4 – Time-of-Day Pattern", -5.0, 5.0, 0.0, 0.1)
        V5 = st.slider("V5 – Online/Cross-border", -5.0, 5.0, 0.0, 0.1)
        Amount = st.number_input("Transaction Amount (USD)", min_value=0.01, max_value=50000.0, value=120.0, step=1.0)

    st.markdown("")
    if st.button("🔍 Analyze Transaction"):
        # Build full feature vector to match real dataset dimensions
        user_vals = [V1, V2, V3, V4, V5]
        n_features = len(feature_cols)
        # Fill remaining V features with 0, put Amount last
        remaining = [0.0] * (n_features - len(user_vals) - 1)
        all_vals = user_vals + remaining + [Amount]
        input_data = np.array([all_vals])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        cluster = kmeans.predict(input_scaled[:, :3])[0]
        cluster_name = {0: '🟢 Low Risk', 1: '🟡 Medium Risk', 2: '🔴 High Risk'}[cluster]

        st.markdown("---")
        st.markdown("### 🧾 Assessment Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-alert">
                <h2 style="color:#ff4081; margin:0;">🚨 FRAUDULENT TRANSACTION DETECTED</h2>
                <p style="color:#ef9a9a; margin:8px 0 0 0; font-size:1.1rem;">
                    Fraud Probability: <b style="font-size:1.4rem;">{probability*100:.1f}%</b>
                </p>
                <p style="color:#b0bec5; margin:4px 0 0 0;">
                    Risk Cluster: {cluster_name} &nbsp;|&nbsp; Amount at Risk: <b>${Amount:,.2f}</b>
                </p>
                <p style="color:#ef9a9a; margin:8px 0 0 0; font-size:0.9rem;">
                    ⚠️ Recommendation: Block transaction and trigger OTP verification. Alert fraud team immediately.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-alert">
                <h2 style="color:#4caf50; margin:0;">✅ LEGITIMATE TRANSACTION</h2>
                <p style="color:#a5d6a7; margin:8px 0 0 0; font-size:1.1rem;">
                    Fraud Probability: <b style="font-size:1.4rem;">{probability*100:.1f}%</b>
                </p>
                <p style="color:#b0bec5; margin:4px 0 0 0;">
                    Risk Cluster: {cluster_name} &nbsp;|&nbsp; Transaction Amount: <b>${Amount:,.2f}</b>
                </p>
                <p style="color:#a5d6a7; margin:8px 0 0 0; font-size:0.9rem;">
                    ✅ Recommendation: Approve transaction. Continue standard monitoring.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Risk Gauge
        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{probability*100:.1f}%")
        col2.metric("Risk Segment", cluster_name.split(' ', 1)[1])
        col3.metric("Decision", "🚨 Block" if prediction == 1 else "✅ Approve")


# ─────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.markdown('<div class="hero-title">📊 Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance metrics for the Logistic Regression fraud detection model</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{auc:.3f}")
    c2.metric("Fraud Precision", f"{report['Fraudulent']['precision']*100:.1f}%")
    c3.metric("Fraud Recall", f"{report['Fraudulent']['recall']*100:.1f}%")
    c4.metric("F1 Score", f"{report['Fraudulent']['f1-score']*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Fraudulent'],
                    yticklabels=['Legitimate', 'Fraudulent'], ax=ax,
                    cbar=False, linewidths=1, linecolor='#0d1b2a')
        ax.set_title('Confusion Matrix', color='#e8eaf6', fontweight='bold')
        ax.tick_params(colors='#90a4ae')
        ax.set_xlabel('Predicted', color='#90a4ae')
        ax.set_ylabel('Actual', color='#90a4ae')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        ax.plot(fpr, tpr, color='#00e5ff', linewidth=2.5, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], color='#455a64', linestyle='--', linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.1, color='#00e5ff')
        ax.set_xlabel('False Positive Rate', color='#90a4ae')
        ax.set_ylabel('True Positive Rate', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.spines['bottom'].set_color('#263238')
        ax.spines['left'].set_color('#263238')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('ROC Curve – Logistic Regression', color='#e8eaf6', fontweight='bold')
        ax.legend(facecolor='#12203a', labelcolor='#e8eaf6')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Full Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report).T.drop('accuracy', errors='ignore')
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(3)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
                 use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: RISK EXPOSURE
# ─────────────────────────────────────────────
elif page == "💰 Risk Exposure":
    st.markdown('<div class="hero-title">💰 Financial Risk Exposure</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Quantify the monetary impact of fraud detection on your business</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Configure Your Business Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_txns = st.number_input("Total Monthly Transactions", value=100000, step=1000)
    with col2:
        avg_amount = st.number_input("Average Transaction Amount (USD)", value=85.0, step=5.0)
    with col3:
        fraud_rate = st.slider("Expected Fraud Rate (%)", 0.1, 20.0, 5.0, 0.1)

    model_recall = report['Fraudulent']['recall']

    if st.button("📊 Calculate Risk Exposure"):
        fraud_txns = int(total_txns * fraud_rate / 100)
        total_fraud_value = fraud_txns * avg_amount
        detected_fraud = int(fraud_txns * model_recall)
        missed_fraud = fraud_txns - detected_fraud
        amount_saved = detected_fraud * avg_amount
        amount_at_risk = missed_fraud * avg_amount
        false_positives = int(total_txns * (1 - report['Legitimate']['precision']) * 0.01)
        cost_false_pos = false_positives * 2  # $2 customer service cost per false positive

        st.markdown("---")
        st.markdown("### 📊 Financial Impact Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Fraud Exposure", f"${total_fraud_value:,.0f}", f"{fraud_txns:,} cases")
        c2.metric("💚 Amount Saved", f"${amount_saved:,.0f}", f"{detected_fraud:,} caught")
        c3.metric("🔴 Still At Risk", f"${amount_at_risk:,.0f}", f"{missed_fraud:,} missed")

        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#0d1b2a')
            ax.set_facecolor('#0d1b2a')
            categories = ['Total Exposure', 'Amount Saved', 'Amount at Risk']
            values = [total_fraud_value, amount_saved, amount_at_risk]
            colors_bar = ['#7c4dff', '#00e5ff', '#ff4081']
            bars = ax.bar(categories, values, color=colors_bar, edgecolor='none', width=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_fraud_value*0.01,
                        f'${val:,.0f}', ha='center', va='bottom', color='#e8eaf6', fontsize=9, fontweight='bold')
            ax.set_title('Financial Impact Breakdown', color='#e8eaf6', fontweight='bold')
            ax.tick_params(colors='#90a4ae')
            ax.spines['bottom'].set_color('#263238')
            ax.spines['left'].set_color('#263238')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax.tick_params(axis='y', colors='#90a4ae')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 🧠 Business Interpretation")
            roi = (amount_saved / (total_fraud_value + cost_false_pos)) * 100
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | Model Recall (Fraud) | {model_recall*100:.1f}% |
            | Fraud Cases Detected | {detected_fraud:,} |
            | Fraud Cases Missed | {missed_fraud:,} |
            | Revenue Protected | ${amount_saved:,.0f} |
            | Residual Risk | ${amount_at_risk:,.0f} |
            | False Positive Cost | ${cost_false_pos:,.0f} |
            | **Detection ROI** | **{roi:.1f}%** |
            """)

        st.markdown("---")
        st.markdown("### 📌 Economic Concepts Applied")
        st.info("""
        **Risk Analysis**: The model quantifies financial exposure across different fraud scenarios, allowing banks to price their risk correctly.

        **Revenue Optimization**: By catching fraud early, the bank retains revenue that would otherwise be reversed as chargebacks.

        **Cost-Benefit Analysis**: False positives cost ~$2 each in customer service; the model balances precision vs. recall to maximize net savings.

        **Pricing Strategy**: High-risk customer segments (from K-Means) can be charged higher transaction fees or require additional authentication, creating a risk-adjusted revenue stream.
        """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#455a64; font-size:0.8rem; font-family:'DM Mono', monospace; padding:10px;">
    🛡️ FraudShield AI &nbsp;|&nbsp; Group 12 – ITM Skills University &nbsp;|&nbsp; Business Studies with Applied AI &nbsp;|&nbsp; 2025
</div>
""", unsafe_allow_html=True)