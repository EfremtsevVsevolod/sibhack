import os
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import run_model

MAIN_MODEL_FILE = "model/main_model.pickle"
MAIN_SCALER_FILE = "model/main_scaler.pickle"
TRANSFORMER_FILE = "model/transformer.pickle"
SUB_MODEL_FILE = "model/subject_inf_rate_model.pickle"
SUB_SCALER_FILE = "model/sub_scaler.pickle"
SAVE_DATASET_FILE = "cache/save_data_file.csv"
RATE = (("Низкий", 1.0), ("Средний", 2.5))


@st.experimental_singleton()
def load_model():
    """
    Loading model for pandemic prediction
    """
    with open(MAIN_MODEL_FILE, "rb") as file:
        main_model = pickle.load(file)

    with open(MAIN_SCALER_FILE, "rb") as file:
        main_scaler = pickle.load(file)

    with open(TRANSFORMER_FILE, "rb") as file:
        transformer = pickle.load(file)

    with open(SUB_MODEL_FILE, "rb") as file:
        sub_model = pickle.load(file)

    with open(SUB_SCALER_FILE, "rb") as file:
        sub_scaler = pickle.load(file)

    return main_model, main_scaler, transformer, sub_model, sub_scaler


def save_data(dataset):
    """Saving received predictions"""
    if os.path.exists(SAVE_DATASET_FILE):
        save_dataset = pd.read_csv(SAVE_DATASET_FILE)
        save_dataset = pd.merge(save_dataset, dataset)
    else:
        save_dataset = dataset

    save_dataset.to_csv(SAVE_DATASET_FILE, index=False)


def visualisation(vis, column, count, ascending, title, ax):
    vis = vis.groupby(by=[column])["inf_rate"].mean().sort_values(ascending=ascending).reset_index().head(count)
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("pastel")
    sns.barplot(x="inf_rate", y=column, data=vis, color="r", ax=ax)
    ax.set(title= title,xlabel='Риск заражения', ylabel='')
    sns.despine(left=True, bottom=True)


def subject_visualisation(vis, column, top_infec, top_non_infec, headers):
    if top_infec + top_non_infec > 0:
        st.header(headers[0])
        col1, col2 = st.columns(2)

        with col1:
            if top_infec > 0:
                fig1, ax1 = plt.subplots(figsize=(3, top_infec * 0.3))
                visualisation(vis, column, top_infec, False, headers[1], ax1)
                st.pyplot(fig1)

        with col2:
            if top_non_infec > 0:
                fig2, ax2 = plt.subplots(figsize=(3, top_non_infec * 0.3))
                visualisation(vis, column, top_non_infec, True, headers[2], ax2)
                st.pyplot(fig2)


def main():
    st.title("Предсказание распространения пандемии по городам России")

    main_model, main_scaler, transformer, sub_model, sub_scaler = load_model()
    st.sidebar.title("Опции")
    vis_data = None

    uploaded_data_file = st.sidebar.file_uploader("Выберите файл с данными", type="csv")

    if uploaded_data_file is not None:
        dataset = pd.read_csv(uploaded_data_file).drop(columns=["Unnamed: 0"])
        vis_data, result = run_model(dataset, main_model, main_scaler, transformer, sub_model, sub_scaler)
        save_data(vis_data)

    if vis_data is None and os.path.exists(SAVE_DATASET_FILE):
        vis_data = pd.read_csv(SAVE_DATASET_FILE)
    
    st.text(len(vis_data))
    
    if vis_data is not None:
        city_req = st.sidebar.text_input("Поиск города")
        top_subject_infec = st.sidebar.slider("Топ областей с высоким уровнем опасности заражения", 0, min(len(vis_data), 30), 0, 1)
        top_subject_non_infec = st.sidebar.slider("Топ областей с низким уровнем опасности заражения", 0, min(len(vis_data), 30), 0, 1)
        top_district_infec = st.sidebar.slider("Топ округов с высоким уровнем опасности заражения", 0, min(len(vis_data), 8), 0, 1)
        top_district_non_infec = st.sidebar.slider("Топ округов с низким уровнем опасности заражения", 0, min(len(vis_data), 8), 0, 1)

        if city_req != "":
            if city_req in set(vis_data["name"]):
                city_inf_rate = float(vis_data.query(f"name == '{city_req}'")["inf_rate"].iloc[0])
                rate = "Высокий"
                for rate_name, rate_border in RATE:
                    if city_inf_rate < rate_border:
                        rate = rate_name
                        break

                st.markdown(fr"#### В городе {city_req} риск заражения $-$ {city_inf_rate:.3f} ({rate})")
            else:
                st.text(fr"#### В базе нет информации о вашем городе. Попробуйте другой.")

        subject_visualisation(vis_data, "subject", top_subject_infec, top_subject_non_infec, ("Статистика по субъектам", "Опасные", "Безопасные"))
        subject_visualisation(vis_data, "district", top_district_infec, top_district_non_infec, ("Статистика по округам", "Опасные", "Безопасные"))


if __name__ == "__main__":
    main()
