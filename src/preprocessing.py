#!c1.8
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor


def drop_defect_rows(dataset, drop=False, unique=False):
    dataset = dataset.copy()

    drops = [
        "Алушта", "Евпатория", "Керчь"
    ]

    uniques = [
        ("Белогорск", "Южный"), ("Благовещенск", "Приволжский"),
        ("Гурьевск", "Сибирский"), ("Заречный", "Приволжский"),
        ("Киров", "Приволжский"), ("Красноармейск", "Приволжский"),
        ("Краснослободск", "Приволжский")
    ]

    dataset = dataset.drop_duplicates()

    if drop:
        dataset = dataset.query(f"name not in {drops}")
    else:
        for drop in drops:
            drop_rows = dataset.query(f"name == '{drop}'")
            inf_rate = np.mean(drop_rows["inf_rate"])
            drop_rows = drop_rows.iloc[0]
            drop_rows["inf_rate"] = inf_rate
            dataset = dataset.query(f"name != '{drop}'")
            dataset = dataset.append(drop_rows, ignore_index=True)

    if unique:
        for name, district in uniques:
            dataset = dataset.query(f"name != '{name}' or district != '{district}'")

    return dataset


def transform_cat_features(dataset_input, transformer):
    dataset = dataset_input.copy()

    transformed = transformer.transform(dataset).toarray()
    new_columns = transformer.get_feature_names_out().tolist()

    for i in range(len(new_columns)):
        dataset[new_columns[i]] = transformed[:, i]

    dataset = dataset.drop(columns=['district', 'subject'])

    return dataset


def preprocessing(covid_data):
    covid_data = covid_data.copy()

    covid_data = drop_defect_rows(covid_data, drop=True, unique=True)
    visualization_columns = covid_data[["name", "district", "subject"]]
    covid_data = covid_data.drop(columns=["name", "region_x"])
    covid_data["has_metro"] = covid_data["has_metro"].astype(int)

    columns_is_nan = ['children_places', 'cleanness', 'ecology', 'ekmo_number',
       'ekmo_per_100k', 'epirank_avia', 'epirank_avia_cat', 'epirank_bus',
       'epirank_bus_cat', 'epirank_train', 'epirank_train_cat',
       'ivl_number', 'ivl_per_100k', 'life_costs',
       'life_quality_place_rating', 'neighbourhood',
       'num_patients_tubercul_1992', 'num_patients_tubercul_1993',
       'num_patients_tubercul_1994', 'num_patients_tubercul_1995',
       'num_patients_tubercul_1996', 'num_patients_tubercul_1997',
       'num_patients_tubercul_1998', 'num_patients_tubercul_1999',
       'num_patients_tubercul_2000', 'num_patients_tubercul_2001',
       'num_patients_tubercul_2002', 'num_patients_tubercul_2003',
       'num_patients_tubercul_2004', 'num_patients_tubercul_2005',
       'num_patients_tubercul_2006', 'num_patients_tubercul_2007',
       'num_patients_tubercul_2008', 'num_patients_tubercul_2009',
       'num_patients_tubercul_2010', 'num_patients_tubercul_2011',
       'num_patients_tubercul_2012', 'num_patients_tubercul_2013',
       'public_services', 'public_transport', 'security',
       'shops_and_malls', 'sport_and_outdoor']

    covid_data = covid_data.drop(columns=columns_is_nan)

    return visualization_columns, covid_data
