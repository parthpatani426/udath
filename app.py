import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Autism Screening Analysis",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ───────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #4A90D9;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin: 0.3rem;
    }
    .metric-val { font-size: 2rem; font-weight: bold; }
    .metric-lbl { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #333;
        border-left: 5px solid #4A90D9; padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    .algo-badge {
        display: inline-block; background: #4A90D9; color: white;
        border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.85rem; margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── LOAD DATA ────────────────────────────
@st.cache_data
def load_data():
    df_child = pd.read_excel("Autism_Child_data_in_excel.xlsx")
    df_screen = pd.read_csv("autism_screening.csv")
    return df_child, df_screen

@st.cache_data
def prepare_data(df_child, df_screen):
    # ── Child dataset ──
    dc = df_child.copy()
    dc.columns = dc.columns.str.strip()
    le = LabelEncoder()
    cat_cols = dc.select_dtypes(include='object').columns
    for col in cat_cols:
        dc[col] = dc[col].astype(str).str.strip().str.strip("'")
        dc[col] = le.fit_transform(dc[col])
    X_child = dc.drop(columns=['Class'])
    y_child = dc['Class']

    # ── Screening dataset ──
    ds = df_screen.copy()
    ds['age'] = ds['age'].fillna(ds['age'].median())
    ds['result'] = ds['result'].fillna(ds['result'].median())
    cat_cols2 = ds.select_dtypes(include='object').columns
    for col in cat_cols2:
        ds[col] = ds[col].astype(str).str.strip()
        ds[col] = le.fit_transform(ds[col])
    X_screen = ds.drop(columns=['Class/ASD'])
    y_screen = ds['Class/ASD']

    return X_child, y_child, X_screen, y_screen

@st.cache_data
def run_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "Naive Bayes":         GaussianNB()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": model, "acc": acc, "cm": cm,
            "fpr": fpr, "tpr": tpr, "auc": roc_auc,
            "report": report, "y_test": y_test, "y_pred": y_pred
        }
    return results, X_train, X_test, y_train, y_test

# ──────────────────────────── SIDEBAR ────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Autism_spectrum_infinity_awareness_symbol.svg/240px-Autism_spectrum_infinity_awareness_symbol.svg.png", width=80)
st.sidebar.markdown("## 🧩 Autism Screening\n**ML Analysis Dashboard**")
st.sidebar.markdown("---")

dataset_choice = st.sidebar.radio(
    "Select Dataset",
    ["Child Data (Excel)", "Screening Data (CSV)", "Compare Both"])

page = st.sidebar.selectbox(
    "📑 Navigate to",
    ["🏠 Overview & EDA",
     "📊 Data Visualizations",
     "🤖 ML Models (5 Algorithms)",
     "📈 Model Comparison",
     "🔍 Predict New Case"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Project by:** Diploma Computer Engineering\n\n**Company:** HMIES Solutions")

# ─────────────────────────── LOAD ───────────────────────────────
df_child, df_screen = load_data()
X_child, y_child, X_screen, y_screen = prepare_data(df_child, df_screen)

if dataset_choice == "Child Data (Excel)":
    X, y, df_raw = X_child, y_child, df_child
    dataset_label = "Child Autism Data"
elif dataset_choice == "Screening Data (CSV)":
    X, y, df_raw = X_screen, y_screen, df_screen
    dataset_label = "Autism Screening Data"
else:
    X, y, df_raw = X_child, y_child, df_child
    dataset_label = "Child Autism Data (default for models)"

results, X_train, X_test, y_train, y_test = run_all_models(X, y)

# ══════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════
if page == "🏠 Overview & EDA":
    st.markdown('<div class="main-title">🧩 Autism Spectrum Disorder — ML Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Industrial Training Project | HMIES Solutions | Diploma Computer Engineering</div>', unsafe_allow_html=True)

    st.markdown("### 📂 Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{df_child.shape[0]}</div><div class="metric-lbl">Child Records</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{df_screen.shape[0]}</div><div class="metric-lbl">Screening Records</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{df_child.shape[1]}</div><div class="metric-lbl">Features</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">5</div><div class="metric-lbl">ML Algorithms</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Child Data (Excel)", "Screening Data (CSV)"])
    with tab1:
        st.dataframe(df_child.head(10), use_container_width=True)
        st.write(f"**Shape:** {df_child.shape[0]} rows × {df_child.shape[1]} columns")
    with tab2:
        st.dataframe(df_screen.head(10), use_container_width=True)
        st.write(f"**Shape:** {df_screen.shape[0]} rows × {df_screen.shape[1]} columns")

    st.markdown('<div class="section-header">📊 Statistical Description</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown('<div class="section-header">🤖 Algorithms Used</div>', unsafe_allow_html=True)
    algo_desc = {
        "Logistic Regression": "Predicts probability of autism (YES/NO) using a sigmoid function.",
        "Decision Tree": "Builds a tree of if-else decisions to classify autism.",
        "Random Forest": "Combines 100 decision trees for higher accuracy.",
        "SVM": "Finds the best boundary (hyperplane) to separate autism classes.",
        "Naive Bayes": "Uses probability calculations based on Bayes theorem."
    }
    for algo, desc in algo_desc.items():
        st.markdown(f'<span class="algo-badge">✅ {algo}</span> — {desc}', unsafe_allow_html=True)
        st.write("")

# ══════════════════════════════════════════════════
#  PAGE 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════
elif page == "📊 Data Visualizations":
    st.markdown('<div class="main-title">📊 Data Visualizations</div>', unsafe_allow_html=True)

    df = df_child  # always use child data for visuals

    # 1. Class Distribution
    st.markdown('<div class="section-header">1. Autism Class Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        class_counts = df['Class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        fig = px.bar(class_counts, x='Class', y='Count',
                     color='Class', title="Autism Cases (YES vs NO)",
                     color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(class_counts, names='Class', values='Count',
                      title="Proportion of Autism Cases",
                      color='Class',
                      color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
        st.plotly_chart(fig2, use_container_width=True)

    # 2. Age Distribution
    st.markdown('<div class="section-header">2. Age Distribution</div>', unsafe_allow_html=True)
    fig3 = px.histogram(df, x='age', color='Class', nbins=20,
                        title="Age Distribution by Autism Class",
                        barmode='overlay',
                        color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
    st.plotly_chart(fig3, use_container_width=True)

    # 3. Gender Distribution
    st.markdown('<div class="section-header">3. Gender vs Autism</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender_class = df.groupby(['gender', 'Class']).size().reset_index(name='count')
        fig4 = px.bar(gender_class, x='gender', y='count', color='Class',
                      title="Gender Distribution by Autism Class", barmode='group',
                      color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
        st.plotly_chart(fig4, use_container_width=True)
    with col2:
        fig5 = px.pie(df, names='gender', title="Gender Split in Dataset",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig5, use_container_width=True)

    # 4. Score Analysis
    st.markdown('<div class="section-header">4. AQ-10 Screening Score Analysis</div>', unsafe_allow_html=True)
    score_cols = [f'A{i}_Score' for i in range(1, 11)]
    score_means = df.groupby('Class')[score_cols].mean().T.reset_index()
    score_means.columns = ['Question', 'NO', 'YES']
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name='NO (Not Autism)', x=score_means['Question'], y=score_means['NO'],
                          marker_color='#2ECC71'))
    fig6.add_trace(go.Bar(name='YES (Autism)', x=score_means['Question'], y=score_means['YES'],
                          marker_color='#E74C3C'))
    fig6.update_layout(title="Average AQ-10 Score per Question by Autism Class",
                       barmode='group', xaxis_title="Question", yaxis_title="Avg Score")
    st.plotly_chart(fig6, use_container_width=True)

    # 5. Heatmap
    st.markdown('<div class="section-header">5. Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = df[score_cols + ['age', 'result']].corr()
    fig7 = px.imshow(num_cols, title="Feature Correlation Heatmap",
                     color_continuous_scale='RdBu_r', text_auto=True)
    st.plotly_chart(fig7, use_container_width=True)

    # 6. Scatter: Age vs Result
    st.markdown('<div class="section-header">6. Age vs Total Score (Scatter Plot)</div>', unsafe_allow_html=True)
    fig8 = px.scatter(df, x='age', y='result', color='Class',
                      title="Age vs AQ Score — Coloured by Autism Class",
                      color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'},
                      hover_data=['gender'])
    st.plotly_chart(fig8, use_container_width=True)

    # 7. Jaundice & Family history
    st.markdown('<div class="section-header">7. Jaundice & Family History Impact</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        j_class = df.groupby(['jundice', 'Class']).size().reset_index(name='count')
        fig9 = px.bar(j_class, x='jundice', y='count', color='Class',
                      title="Jaundice History vs Autism", barmode='group',
                      color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
        st.plotly_chart(fig9, use_container_width=True)
    with col2:
        a_class = df.groupby(['autism', 'Class']).size().reset_index(name='count')
        fig10 = px.bar(a_class, x='autism', y='count', color='Class',
                       title="Family Autism History vs Autism", barmode='group',
                       color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
        st.plotly_chart(fig10, use_container_width=True)

    # 8. Ethnicity
    st.markdown('<div class="section-header">8. Ethnicity Distribution</div>', unsafe_allow_html=True)
    eth = df['ethnicity'].str.strip().str.strip("'").value_counts().reset_index()
    eth.columns = ['Ethnicity', 'Count']
    fig11 = px.bar(eth, x='Count', y='Ethnicity', orientation='h',
                   title="Ethnicity of Participants",
                   color='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig11, use_container_width=True)

    # 9. Box plot
    st.markdown('<div class="section-header">9. Score Distribution Box Plot</div>', unsafe_allow_html=True)
    df_melt = df[score_cols + ['Class']].melt(id_vars='Class', var_name='Question', value_name='Score')
    fig12 = px.box(df_melt, x='Question', y='Score', color='Class',
                   title="Score Distribution per Question",
                   color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'})
    st.plotly_chart(fig12, use_container_width=True)

    # 10. Line graph: cumulative result by age
    st.markdown('<div class="section-header">10. Average Score by Age (Line Graph)</div>', unsafe_allow_html=True)
    age_result = df.groupby('age')['result'].mean().reset_index()
    fig13 = px.line(age_result, x='age', y='result',
                    title="Average AQ Score Across Ages",
                    markers=True)
    st.plotly_chart(fig13, use_container_width=True)

# ══════════════════════════════════════════════════
#  PAGE 3 — ML MODELS
# ══════════════════════════════════════════════════
elif page == "🤖 ML Models (5 Algorithms)":
    st.markdown('<div class="main-title">🤖 5 Machine Learning Algorithms</div>', unsafe_allow_html=True)
    st.info(f"Training on: **{dataset_label}** | Train split: 80% | Test split: 20%")

    algo_name = st.selectbox(
        "Select Algorithm to Inspect",
        list(results.keys()))

    res = results[algo_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Accuracy", f"{res['acc']*100:.2f}%")
    with col2:
        prec = res['report']['weighted avg']['precision']
        st.metric("🎯 Precision", f"{prec:.2f}")
    with col3:
        recall = res['report']['weighted avg']['recall']
        st.metric("📡 Recall", f"{recall:.2f}")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_df = pd.DataFrame(
            res['cm'],
            index=['Actual NO', 'Actual YES'],
            columns=['Predicted NO', 'Predicted YES'])
        fig_cm = px.imshow(cm_df, text_auto=True, title=f"Confusion Matrix — {algo_name}",
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'],
                                      name=f"AUC = {res['auc']:.3f}",
                                      line=dict(color='#4A90D9', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                      line=dict(color='gray', dash='dash'),
                                      name="Random Classifier"))
        fig_roc.update_layout(title=f"ROC Curve — {algo_name}",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    # Classification Report
    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(res['report']).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # Feature Importance (for tree-based models)
    if algo_name in ["Decision Tree", "Random Forest"]:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame({
            'Feature': X.columns,
            'Importance': res['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                        title=f"Top Feature Importances — {algo_name}",
                        color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════
#  PAGE 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<div class="main-title">📈 Model Comparison Dashboard</div>', unsafe_allow_html=True)

    # Accuracy bar chart
    st.markdown('<div class="section-header">1. Accuracy Comparison</div>', unsafe_allow_html=True)
    acc_df = pd.DataFrame([
        {'Algorithm': name, 'Accuracy': r['acc']*100, 'AUC': r['auc']}
        for name, r in results.items()
    ]).sort_values('Accuracy', ascending=False)

    fig_acc = px.bar(acc_df, x='Algorithm', y='Accuracy',
                     color='Accuracy', title="Accuracy of All 5 Algorithms (%)",
                     color_continuous_scale='Viridis', text='Accuracy')
    fig_acc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_acc, use_container_width=True)

    # AUC comparison
    st.markdown('<div class="section-header">2. AUC Score Comparison</div>', unsafe_allow_html=True)
    fig_auc = px.bar(acc_df, x='Algorithm', y='AUC',
                     color='AUC', title="AUC Score of All 5 Algorithms",
                     color_continuous_scale='Plasma', text='AUC')
    fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_auc, use_container_width=True)

    # All ROC curves together
    st.markdown('<div class="section-header">3. All ROC Curves Together</div>', unsafe_allow_html=True)
    fig_rocs = go.Figure()
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    for (name, r), color in zip(results.items(), colors):
        fig_rocs.add_trace(go.Scatter(
            x=r['fpr'], y=r['tpr'],
            name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=color, width=2)))
    fig_rocs.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                   line=dict(color='gray', dash='dash'),
                                   name="Random"))
    fig_rocs.update_layout(title="ROC Curves — All 5 Algorithms",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate")
    st.plotly_chart(fig_rocs, use_container_width=True)

    # Precision / Recall / F1 radar
    st.markdown('<div class="section-header">4. Precision, Recall, F1 Comparison</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame([{
        'Algorithm': name,
        'Precision': r['report']['weighted avg']['precision'],
        'Recall':    r['report']['weighted avg']['recall'],
        'F1-Score':  r['report']['weighted avg']['f1-score'],
        'Accuracy':  r['acc']
    } for name, r in results.items()])
    fig_metrics = go.Figure()
    for _, row in metrics_df.iterrows():
        fig_metrics.add_trace(go.Scatterpolar(
            r=[row['Precision'], row['Recall'], row['F1-Score'], row['Accuracy']],
            theta=['Precision', 'Recall', 'F1-Score', 'Accuracy'],
            fill='toself', name=row['Algorithm']))
    fig_metrics.update_layout(title="Radar Chart — Algorithm Performance",
                               polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">5. Full Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df.set_index('Algorithm').style.highlight_max(color='lightgreen').format("{:.4f}"),
                 use_container_width=True)

    best = acc_df.iloc[0]
    st.success(f"🏆 **Best Algorithm: {best['Algorithm']}** with Accuracy = {best['Accuracy']:.2f}%")

# ══════════════════════════════════════════════════
#  PAGE 5 — PREDICT
# ══════════════════════════════════════════════════
elif page == "🔍 Predict New Case":
    st.markdown('<div class="main-title">🔍 Predict Autism for a New Child</div>', unsafe_allow_html=True)
    st.info("Fill in the AQ-10 questionnaire and demographic info to predict autism.")

    st.markdown("### AQ-10 Screening Questions (0 = No / 1 = Yes)")
    cols = st.columns(5)
    scores = []
    questions = [
        "Does the child look at you when called?",
        "Is eye contact easy for the child?",
        "Does the child point to show interest?",
        "Does the child pretend play?",
        "Does the child follow where you look?",
        "Does the child comfort others in distress?",
        "Is the child's first word remembered?",
        "Does the child use simple gestures?",
        "Does the child stare at nothing with no purpose?",
        "Does the child look at your face for reaction?"
    ]
    for i, q in enumerate(questions):
        with cols[i % 5]:
            s = st.selectbox(f"A{i+1}", [0, 1], key=f"q{i}", help=q)
            scores.append(s)

    st.markdown("### Demographics")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age (years)", 1, 17, 5)
    with c2:
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    with c3:
        jaundice = st.selectbox("Jaundice at birth?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    c4, c5 = st.columns(2)
    with c4:
        autism_fam = st.selectbox("Family history of autism?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    with c5:
        algo_choice = st.selectbox("Choose Algorithm", list(results.keys()))

    result_score = sum(scores)

    if st.button("🔮 Predict Now", type="primary"):
        # Build input vector matching X_child columns
        input_dict = {f'A{i+1}_Score': scores[i] for i in range(10)}
        input_dict['age'] = age
        input_dict['gender'] = gender
        input_dict['jundice'] = jaundice
        input_dict['autism'] = autism_fam
        input_dict['result'] = result_score
        # fill remaining columns with 0
        for col in X.columns:
            if col not in input_dict:
                input_dict[col] = 0

        input_df = pd.DataFrame([input_dict])[X.columns]
        model = results[algo_choice]['model']
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        label = "YES — Autism Likely" if pred == 1 else "NO — Autism Unlikely"
        color = "🔴" if pred == 1 else "🟢"

        st.markdown("---")
        st.markdown(f"## {color} Prediction: **{label}**")
        st.markdown(f"**Algorithm used:** {algo_choice}")
        st.markdown(f"**AQ-10 Total Score:** {result_score} / 10")

        conf_df = pd.DataFrame({'Class': ['NO', 'YES'], 'Confidence': prob})
        fig_conf = px.bar(conf_df, x='Class', y='Confidence',
                          color='Class',
                          color_discrete_map={'YES': '#E74C3C', 'NO': '#2ECC71'},
                          title="Prediction Confidence")
        st.plotly_chart(fig_conf, use_container_width=True)

        if result_score >= 6:
            st.warning("⚠️ AQ-10 score ≥ 6 suggests a referral for clinical evaluation is recommended.")
        else:
            st.info("ℹ️ AQ-10 score < 6 — lower likelihood, but consult a specialist if concerned.")
