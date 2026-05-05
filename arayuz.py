import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ses_analizi import (
    EMOTION_LABELS,
    aggregate_feature_importance,
    build_feature_table,
    predict_single_file,
    train_emotion_model,
)


st.set_page_config(
    page_title="Duygu Tanıma ve Özellik Analizi",
    page_icon="🎙️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0f172a 38%, #dfe6f2 38%, #eef3fa 100%);
    }
    .hero {
        color: #eff6ff;
        padding: 24px 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(2,12,27,0.98), rgba(15,23,42,0.98));
        box-shadow: 0 18px 46px rgba(2,6,23,.28);
        margin-bottom: 18px;
    }
    .stSidebar {
        background: #08101c;
    }
    .stSidebar .stTextInput label,
    .stSidebar .stSelectbox label,
    .stSidebar .stButton label,
    .stSidebar .stCaption,
    .stSidebar .stMarkdown,
    .stSidebar p,
    .stSidebar span {
        color: #e5e7eb !important;
    }
    .stButton > button {
        background: #1d4ed8;
        color: #f8fafc;
        border: 1px solid #1e40af;
    }
    .stButton > button:hover {
        background: #1e3a8a;
        border-color: #3b82f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0;">🎙️ Ses Duygu Tanıma</h1>
        <p style="margin:8px 0 0 0; opacity:0.88;">MFCC, pitch, ZCR ve enerji tabanlı özellikler ile 5 sınıflı klasik makine öğrenmesi akışı</p>
    </div>
    """,
    unsafe_allow_html=True,
)

APP_STATE_VERSION = "2026-05-05-v2"


def load_feature_table(metadata_source: str, dataset_root: str) -> pd.DataFrame:
    return build_feature_table(metadata_source, dataset_root)


def render_confusion_matrix(y_true, y_pred):
    labels = EMOTION_LABELS
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Karışıklık Matrisi")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


with st.sidebar:
    st.header("Ayarlar")
    metadata_source = st.text_input("Metadata kaynağı", value="birlesik_metadata.xlsx")
    dataset_root = st.text_input("Ses klasörü", value="Dataset")
    st.caption("Varsayılan olarak birleşik metadata dosyası kullanılır.")
    run_button = st.button("Veri Setini İşle ve Modeli Eğit", type="primary")


if "feature_df" not in st.session_state:
    st.session_state.feature_df = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if st.session_state.get("app_state_version") != APP_STATE_VERSION:
    st.session_state.feature_df = None
    st.session_state.model_info = None
    st.session_state.app_state_version = APP_STATE_VERSION


tab1, tab2 = st.tabs(["🔍 Tek Dosya", "📊 Veri Seti ve Model"])

with tab2:
    st.subheader("Toplu eğitim ve değerlendirme")
    if run_button:
        with st.spinner("Öznitelikler çıkarılıyor ve model eğitiliyor..."):
            feature_df = load_feature_table(metadata_source, dataset_root)
            model_info = train_emotion_model(feature_df)
            valid_df = feature_df[feature_df["Sinif"].isin(EMOTION_LABELS)].copy()
            # Exclude rows where audio file wasn't found
            if "Dosya_Bulundu" in valid_df.columns:
                valid_df = valid_df[valid_df["Dosya_Bulundu"] == True].copy()
            valid_df = valid_df[valid_df[model_info["feature_columns"]].notna().any(axis=1)].copy()
            valid_df[model_info["feature_columns"]] = valid_df[model_info["feature_columns"]].fillna(
                valid_df[model_info["feature_columns"]].median(numeric_only=True)
            )
            valid_df["Tahmin"] = model_info["model"].predict(valid_df[model_info["feature_columns"]])
            valid_df["Dogru_mu"] = valid_df["Tahmin"] == valid_df["Sinif"]

            st.session_state.feature_df = feature_df
            st.session_state.model_info = model_info | {"valid_df": valid_df}

        st.success(f"Model hazır. Test kümesi doğruluğu: %{model_info['accuracy'] * 100:.2f}")

    if st.session_state.model_info is not None:
        model_info = st.session_state.model_info
        feature_df = st.session_state.feature_df
        valid_df = model_info["valid_df"]

        metric_cols = st.columns(3)
        metric_cols[0].metric("Test Kümesi Doğruluğu", f"%{model_info['accuracy'] * 100:.2f}")
        metric_cols[1].metric("Örnek Sayısı", f"{len(valid_df)}")
        metric_cols[2].metric("Öznitelik Sayısı", f"{len(model_info['feature_columns'])}")

        st.markdown("### Sınıf Dağılımı")
        st.bar_chart(feature_df["Sinif"].value_counts().reindex(EMOTION_LABELS).fillna(0))

        st.markdown("### Sınıflandırma Raporu")
        report = classification_report(
            valid_df["Sinif"],
            valid_df["Tahmin"],
            labels=EMOTION_LABELS,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

        st.markdown("### Karışıklık Matrisi")
        st.pyplot(render_confusion_matrix(valid_df["Sinif"], valid_df["Tahmin"]))

        st.markdown("### Özellik Grubu Önemleri")
        importance_df = aggregate_feature_importance(model_info["model"], model_info["feature_columns"])
        st.dataframe(importance_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(importance_df["Feature_Group"], importance_df["Importance"], color="#2563eb")
        ax.invert_yaxis()
        ax.set_xlabel("Önem")
        ax.set_title("Özellik Grup Önemleri")
        st.pyplot(fig)

        st.markdown("### Önizleme")
        st.dataframe(feature_df.head(10), use_container_width=True)
    else:
        st.info("Modeli başlatmak için soldaki butonu kullanın.")


with tab1:
    st.subheader("Tek ses dosyası ile tahmin")
    uploaded = st.file_uploader("WAV dosyası seçin", type=["wav"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.audio(tmp_path)

        with st.spinner("Dosya analiz ediliyor..."):
            if st.session_state.model_info is not None:
                prediction = predict_single_file(
                    tmp_path,
                    model=st.session_state.model_info["model"],
                    feature_cols=st.session_state.model_info["feature_columns"],
                )
            else:
                prediction = predict_single_file(tmp_path)

        os.unlink(tmp_path)

        if "hata" in prediction:
            st.error(prediction["hata"])
        else:
            st.success(f"Tahmin: {prediction['Tahmin'] or 'Model eğitilmedi'}")
            cols = st.columns(4)
            cols[0].metric("Pitch Ortalama", f"{prediction['Pitch_Mean_Hz']}")
            cols[1].metric("Pitch Std Sapma", f"{prediction['Pitch_Std_Hz']}")
            cols[2].metric("ZCR (saniye)", f"{prediction['ZCR_Per_Sec']}")
            cols[3].metric("Enerji", f"{prediction['Energy_Mean']:.6f}" if prediction["Energy_Mean"] is not None else "-")

            st.json(prediction)
    else:
        st.info("Bir WAV dosyası yükleyin ve modelinizle tahmin alın.")
