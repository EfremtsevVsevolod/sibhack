import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .preprocessing import preprocessing, transform_cat_features


def predict_mean_subject_inf_rate(model, scaler, covid_data, num_columns, target_column):
    covid_data = covid_data.groupby('subject')[num_columns + target_column].apply(np.mean)

    print(covid_data.columns)

    data = covid_data.drop(columns=target_column)
    data[data.columns] = scaler.transform(data)
    subject_inf_rate = pd.Series(data=model.predict(data).ravel(), index=data.index, name="inf_rate_subject")

    return subject_inf_rate


def run_model(dataset, main_model, main_scaler, transformer, sub_model, sub_scaler):
    """Running the Model on new dataset"""
    vis_data, covid_data = preprocessing(dataset)

    target_column = ["inf_rate"]
    categories_columns = ["district", "subject", "has_metro"]
    num_columns = covid_data.columns.drop(categories_columns + target_column).tolist()

    subject_inf_rate = predict_mean_subject_inf_rate(sub_model, sub_scaler, covid_data, num_columns, target_column)
    covid_data = covid_data.join(subject_inf_rate, on='subject')

    covid_data = covid_data.drop(columns=target_column)
    data_cat = covid_data[categories_columns]
    data_num = covid_data.drop(columns=categories_columns)

    data_cat = transform_cat_features(data_cat, transformer=transformer)
    data_num = main_scaler.transform(data_num)
    data = np.hstack((data_num, data_cat))

    predict = np.clip(main_model.predict(data), 0.5, 5)
    vis_data["inf_rate"] = predict

    return vis_data, predict
