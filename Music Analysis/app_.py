# ...existing code...
import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor

# try optional deep learning (PyTorch); fallback to sklearn MLP
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# projeto paths
CSV_DEFAULT = "free_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Music Analysis — ML Studio", layout="wide")

# Utilities
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def save_joblib(obj, name):
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(obj, path)
    return path

def download_button_bytes(obj_path, label="Baixar"):
    with open(obj_path, "rb") as f:
        b = f.read()
    st.download_button(label, data=b, file_name=os.path.basename(obj_path))

def scale_data(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if X_test is None:
        return X_train_s, scaler
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

# small torch MLP (if available)
if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=(128,64), task='regression'):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers.append(nn.Linear(prev, h)); layers.append(nn.ReLU()); prev = h
            layers.append(nn.Linear(prev, out_dim))
            if task == 'classification':
                layers.append(nn.LogSoftmax(dim=1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

st.sidebar.title("Configurações")
uploaded = st.sidebar.file_uploader("Carregar free_features.csv (opcional)", type=["csv"])
if uploaded:
    csv_bytes = uploaded.read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
else:
    if os.path.exists(CSV_DEFAULT):
        df = load_csv(CSV_DEFAULT)
    else:
        st.sidebar.error(f"{CSV_DEFAULT} não encontrado. Gere com app.py / analyze.py")
        df = None

task = st.sidebar.selectbox("Tarefa", ["Summary", "ML - Classificação", "ML - Regressão", "Clustering", "Anomalias", "Deep Learning"])
st.sidebar.markdown("---")
st.sidebar.write("Executar")
run_btn = st.sidebar.button("Executar")

if df is None:
    st.stop()

if st.sidebar.checkbox("Mostrar dados brutos"):
    st.dataframe(df.head(200))

numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
st.title("Music Analysis — ML Studio (Melhorado)")
st.write("Interface interativa com modelos tradicionais, regressão linear, deep learning (PyTorch se disponível) e visualizações.")

# Common visualizations
st.header("Visualizações")
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Mapa de correlação")
    if numeric_cols:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, ax=ax, cmap="vlag", center=0, annot=False)
        st.pyplot(fig)
    else:
        st.info("Sem colunas numéricas para correlação.")
with col2:
    st.subheader("Distribuição de feature")
    if numeric_cols:
        sel = st.selectbox("Selecionar coluna para histograma", numeric_cols, index=0)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.histplot(df[sel].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Sem colunas numéricas.")

# Task: Classification
if task == "ML - Classificação":
    st.header("Classificação")
    if not numeric_cols:
        st.error("Nenhuma coluna numérica para treinar.")
    else:
        label_col = st.selectbox("Coluna rótulo", options=[None] + list(df.columns), index=0)
        model_choice = st.selectbox("Modelo", ["RandomForest", "LogisticRegression", "MLP (sklearn)"])
        test_size = st.slider("Test size (%)", 5, 50, 20) / 100.0
        if run_btn:
            if label_col is None:
                st.error("Selecione a coluna de rótulo.")
            else:
                X = df[numeric_cols].copy()
                if label_col in X.columns:
                    X = X.drop(columns=[label_col])
                y = df[label_col].values
                # bin continuous label if needed
                if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 10:
                    y = pd.qcut(y, 3, labels=False)
                X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=42, stratify=(y if len(np.unique(y))>1 else None))
                X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

                if model_choice == "RandomForest":
                    model = RandomForestClassifier(n_estimators=200, random_state=42)
                elif model_choice == "LogisticRegression":
                    model = LogisticRegression(max_iter=2000)
                else:
                    model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)

                with st.spinner("Treinando..."):
                    model.fit(X_train_s, y_train)

                preds = model.predict(X_test_s)
                acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="weighted")
                st.success(f"Accuracy: {acc:.4f} — F1 (weighted): {f1:.4f}")
                st.subheader("Relatório")
                st.text(classification_report(y_test, preds))

                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                if hasattr(model, "feature_importances_"):
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
                    st.subheader("Top features")
                    st.bar_chart(fi)

                pkg = {"model": model, "scaler": scaler, "features": list(X.columns)}
                path = save_joblib(pkg, "classification_model.joblib")
                st.info(f"Modelo salvo: {path}")
                download_button_bytes(path, "Baixar modelo (.joblib)")

# Task: Regression
if task == "ML - Regressão":
    st.header("Regressão")
    if not numeric_cols:
        st.error("Nenhuma coluna numérica para treinar.")
    else:
        target_col = st.selectbox("Coluna alvo", options=[None] + list(df.columns), index=0)
        model_choice = st.selectbox("Modelo", ["RandomForestRegressor", "LinearRegression", "Ridge", "Lasso", "MLPRegressor (sklearn)"])
        test_size = st.slider("Test size (%)", 5, 50, 20) / 100.0
        if run_btn:
            if target_col is None:
                st.error("Selecione a coluna alvo.")
            else:
                X = df[numeric_cols].copy()
                if target_col in X.columns:
                    X = X.drop(columns=[target_col])
                y = df[target_col].values
                X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=test_size, random_state=42)
                X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

                if model_choice == "RandomForestRegressor":
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                elif model_choice == "LinearRegression":
                    model = LinearRegression()
                elif model_choice == "Ridge":
                    model = Ridge(alpha=1.0)
                elif model_choice == "Lasso":
                    model = Lasso(alpha=0.1)
                else:
                    model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=1000)

                with st.spinner("Treinando..."):
                    model.fit(X_train_s, y_train)

                preds = model.predict(X_test_s)
                mse = mean_squared_error(y_test, preds); r2 = r2_score(y_test, preds)
                st.success(f"MSE: {mse:.6f} — R2: {r2:.4f}")

                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(y_test, preds, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("True"); ax.set_ylabel("Predicted")
                st.pyplot(fig)

                pkg = {"model": model, "scaler": scaler, "features": list(X.columns)}
                path = save_joblib(pkg, "regression_model.joblib")
                st.info(f"Modelo salvo: {path}")
                download_button_bytes(path, "Baixar modelo (.joblib)")

# Task: Clustering
if task == "Clustering":
    st.header("Clustering e PCA")
    if not numeric_cols:
        st.error("Sem colunas numéricas.")
    else:
        n_clusters = st.slider("Clusters", 2, 8, 3)
        n_components = st.slider("PCA componentes", 2, min(10, len(numeric_cols)), 2)
        if run_btn:
            X = df[numeric_cols].values
            X_s, scaler = scale_data(X)
            pca = PCA(n_components=n_components, random_state=42)
            X_p = pca.fit_transform(X_s)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_p)
            st.write(pd.Series(labels).value_counts().sort_index())
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(x=X_p[:,0], y=X_p[:,1], hue=labels, palette="tab10", ax=ax)
            st.pyplot(fig)
            out = df.copy(); out["cluster"]=labels
            out_path = "clustered_features.csv"; out.to_csv(out_path, index=False)
            save_joblib({"kmeans": kmeans, "pca": pca, "scaler": scaler, "features": numeric_cols}, "kmeans.joblib")
            st.success(f"Clustering salvo em {out_path}")
            download_button_bytes(out_path, "Baixar clustered_features.csv")

# Task: Anomaly
if task == "Anomalias":
    st.header("Detecção de Anomalias (IsolationForest)")
    contamination = st.slider("Contaminação", 0.01, 0.2, 0.05)
    if run_btn:
        X = df[numeric_cols].values
        X_s, scaler = scale_data(X)
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        with st.spinner("Ajustando..."):
            iso.fit(X_s)
        scores = iso.decision_function(X_s); preds = iso.predict(X_s)
        out = df.copy(); out["anomaly_score"] = -scores; out["anomaly"] = (preds == -1).astype(int)
        st.write("Anomalias:", int(out["anomaly"].sum()))
        st.dataframe(out.sort_values("anomaly_score", ascending=False).head(50))
        out_path = "anomalies.csv"; out.to_csv(out_path, index=False)
        save_joblib({"iso": iso, "scaler": scaler, "features": numeric_cols}, "isolation_forest.joblib")
        st.success(f"Resultados salvos em {out_path}")
        download_button_bytes(out_path, "Baixar anomalies.csv")

# Task: Deep Learning (PyTorch or fallback sklearn MLP)
if task == "Deep Learning":
    st.header("Deep Learning — MLP simples")
    if not numeric_cols:
        st.error("Sem colunas numéricas.")
    else:
        mode = st.selectbox("Modo", ["Regressão", "Classificação"])
        epochs = st.slider("Epochs", 5, 200, 50)
        batch_size = st.selectbox("Batch size", [8,16,32,64], index=2)
        lr = st.number_input("Learning rate", 1e-4, 1e-1, value=1e-3, format="%.4f")
        target_col = st.selectbox("Coluna alvo / label", options=[None] + list(df.columns), index=0)
        if run_btn:
            if target_col is None:
                st.error("Selecione coluna alvo.")
            else:
                X = df[numeric_cols].copy()
                if target_col in X.columns:
                    X = X.drop(columns=[target_col])
                y = df[target_col].values
                # prepare data
                if mode == "Classificação":
                    if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 10:
                        y = pd.qcut(y, 3, labels=False)
                X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=(y if mode=="Classificação" and len(np.unique(y))>1 else None))
                X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

                if TORCH_AVAILABLE:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    in_dim = X_train_s.shape[1]
                    if mode == "Regressão":
                        out_dim = 1
                    else:
                        out_dim = len(np.unique(y_train))
                    model = TorchMLP(in_dim, out_dim, hidden=(128,64), task=('classification' if mode=='Classificação' else 'regression')).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    loss_fn = nn.NLLLoss() if mode=="Classificação" else nn.MSELoss()
                    # training loop
                    losses = []
                    X_t = torch.from_numpy(X_train_s).float().to(device)
                    y_t = torch.from_numpy(np.array(y_train)).long().to(device) if mode=="Classificação" else torch.from_numpy(np.array(y_train)).float().to(device)
                    dataset = torch.utils.data.TensorDataset(X_t, y_t)
                    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    with st.spinner("Treinando (PyTorch)..."):
                        for ep in range(1, epochs+1):
                            epoch_losses = []
                            model.train()
                            for xb, yb in loader:
                                optimizer.zero_grad()
                                out = model(xb)
                                if mode=="Classificação":
                                    loss = loss_fn(out, yb)
                                else:
                                    loss = loss_fn(out.squeeze(), yb)
                                loss.backward(); optimizer.step()
                                epoch_losses.append(loss.item())
                            losses.append(np.mean(epoch_losses))
                            if ep % max(1, epochs//5) == 0:
                                st.write(f"Epoch {ep}/{epochs} — loss: {losses[-1]:.6f}")
                    # evaluate
                    model.eval()
                    with torch.no_grad():
                        X_te = torch.from_numpy(X_test_s).float().to(device)
                        out = model(X_te)
                        if mode=="Classificação":
                            preds = out.argmax(dim=1).cpu().numpy()
                        else:
                            preds = out.squeeze().cpu().numpy()
                    # metrics
                    if mode=="Classificação":
                        acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="weighted")
                        st.success(f"Accuracy: {acc:.4f} — F1: {f1:.4f}")
                        st.text(classification_report(y_test, preds))
                    else:
                        mse = mean_squared_error(y_test, preds); r2 = r2_score(y_test, preds)
                        st.success(f"MSE: {mse:.6f} — R2: {r2:.4f}")
                    # plot loss
                    fig, ax = plt.subplots()
                    ax.plot(losses, label="train_loss")
                    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
                    st.pyplot(fig)
                    # save model state dict
                    path = os.path.join(MODEL_DIR, "torch_mlp.pt")
                    torch.save({"state_dict": model.state_dict(), "scaler": scaler, "cols": numeric_cols}, path)
                    st.info(f"PyTorch model salvo: {path}")
                else:
                    st.info("PyTorch não disponível — usando MLP do scikit-learn")
                    if mode == "Classificação":
                        clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=epochs)
                    else:
                        clf = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=epochs)
                    with st.spinner("Treinando MLP sklearn..."):
                        clf.fit(X_train_s, y_train)
                    preds = clf.predict(X_test_s)
                    if mode=="Classificação":
                        acc = accuracy_score(y_test, preds); f1 = f1_score(y_test, preds, average="weighted")
                        st.success(f"Accuracy: {acc:.4f} — F1: {f1:.4f}")
                        st.text(classification_report(y_test, preds))
                    else:
                        mse = mean_squared_error(y_test, preds); r2 = r2_score(y_test, preds)
                        st.success(f"MSE: {mse:.6f} — R2: {r2:.4f}")
                    path = save_joblib({"model": clf, "scaler": scaler, "features": numeric_cols}, "mlp_sklearn.joblib")
                    st.info(f"Modelo salvo: {path}")
                    download_button_bytes(path, "Baixar modelo (.joblib)")

st.sidebar.markdown("---")
st.sidebar.write("Executar via terminal:")
st.sidebar.code(r'cd "c:\Users\anton\OneDrive\Área de Trabalho\Music Analysis"')
st.sidebar.code("streamlit run streamlit_app.py")
# ...existing code...