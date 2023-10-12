# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Classification

# ## Imports

# +
from datetime import datetime
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    balanced_accuracy_score
)
from sklearn.tree import export_graphviz
import graphviz
from xgboost import XGBClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Option to display all columns of dataframe:
pd.set_option("display.max_columns", None)
# -

# ## Config

# +
SETUP_NAME = 'cls_weighted'

MONTHS = 6

# ---------------------------------------

learning_setup = {
    'data_path': "../data/poc_data.csv",
    'model_export_path': f'./models/{SETUP_NAME}.pkl',
    'model_config_export_path': f'./models/{SETUP_NAME}_config.json',
    'train_from_date': '01/01/2018',
    'label_column': f'resigned_in_{MONTHS}m',
    'validation_size': 0.2,
    'test_size': 0.1,
    'data_weighting': True
}
# -

# -----
# ## Renaming columns from norwegian to english

nor_to_eng_names = {
    "CUST": "customer",
    "År-Mnd": "export_date",
    "Arbeidsgiver nr": "employer_number",
    "PERSON ID": "person_id",
    "Kjønn": "sex",
    "Fødselsdato": "birth_date",
    "Alder": "age",
    "Nationalitet": "nationality",
    "Sivilstatus": "civil_status",
    "Første gang ansatt dato": "first_employeed_date",
    "Sluttdato person": "stop_date_person",
    "Sluttårsak kode": "termination_code",
    "Sluttårsak kode navn": "termination_code_desc",
    "Adresse 1": "address_1",
    "Adresse 2": "address_2",
    "Adresse 3": "address_3",
    "Postnummer ved lønnskjøring": "post_code",
    "Poststed ved lønnskjøring": "post_office",
    "ARBEIDSFORHOLD ID": "employment_id",
    "Arbfnr": "employment_number",
    "Enhet nr": "org_unit",
    "Lokasjon Post nr": "org_unit_post_code",
    "Lokasjon Post navn": "org_unit_post_office",
    "Lokasjon Kommune nr": "org_unit_municipality",
    "Lokasjon Kommune navn": "org_unit_municipality_name",
    "Startdato": "start_date",
    "Gjelder fra": "valid_from",
    "Gjelder til": "valid_to",
    "Sluttdato": "end_date_employment",
    "Status": "status",
    "Ansettelsesform (A-mel.)": "employment_form",
    "Arbeidsforhold type (A-mel.)": "employment_type",
    "Arbeidstidsordning (A-mel.)": "worktime_arrangement",
    "Sluttårsak (A-mel.)": "termination_reason",
    "Ansattform": "employment_type_code",
    "Ansattform tekst": "employment_type_name",
    "Ansattandel": "employed_percentage",
    "Utbetalingsprosent": "payout_percentage",
    "Tilstedeprosent": "attendance_pecentage",
    "Timer per uke": "hours_per_week",
    "Stillingsgruppe kode": "position_group_code",
    "Stillingsgruppe navn": "position_group_name",
    "Stillingskode": "position_code",
    "Stillingskode tekst": "position_name",
    "Stillingsbetegnelse": "position_description",
    "Årsakskode kode": "reason_for_change_code",
    "Årsakskode beskrivelse": "reason_for_change_text",
    "Årslønn": "salary",
    "Antall": "export_code"}


# ## Evaulation metrics

# +
def plot_conf_matrix(y_val, y_pred, title):
    """
    Confusion matrix plot.
    :param y_val: validation label
    :param y_pred: predicted label
    """
    cf_matrix = confusion_matrix(y_val, y_pred)
    
    group_names = ['True Negative\n("correct rejection")\n', 'False Positive\n("false alarm")\n', 'False Negative\n("miss")\n', 'True Positive\n("hit")\n']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    fig, ax = plt.subplots(figsize=(9,7))
    ax.set_title(f'\nConfusion Matrix\n({title})', fontsize=16, pad=30, fontweight='bold', loc='center')
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', annot_kws={'fontsize': 14})
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

def plot_pr_curve(y_val, y_pred, title):
    """
    Precision-Recall curve plot.
    :param y_val: validation label
    :param y_pred: predicted label
    """
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(8.5,5))
    ax.plot(recall, precision, color='purple')
    ax.set_title(f'Precision-Recall Curve\n({title})\n', fontsize=16, pad=5, fontweight='bold', loc='center')
    ax.set_ylabel('Precision (Positive predictive value)', fontsize=14, labelpad=20)
    ax.set_xlabel('Recall (Sensitivity)',  fontsize=14, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

def plot_score(y_val, y_pred, title):
    """
    Classification score plot.
    :param y_val: validation label
    :param y_pred: predicted label
    """
    header = ['Metric','Score']
    data = [['Balanced accuracy', str(round(balanced_accuracy_score(y_val, y_pred), 3))],
            ['Precision', str(round(precision_score(y_val, y_pred), 3))],
            ['Recall', str(round(recall_score(y_val, y_pred), 3))],
            ['F1', str(round(f1_score(y_val, y_pred), 3))]]
    
    fig, ax = plt.subplots(figsize=(7,5))
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.axis('off')
    ax.set_title(f'\nClassification Score Table\n({title})', fontsize=16, pad=5, fontweight='bold', loc='center')
    table = ax.table(cellText=data, colLabels=header, loc='center', cellLoc='center', rowLoc='center')
    table.get_celld()[(0,0)].set_facecolor("#56b5fd")
    table.get_celld()[(0,1)].set_facecolor("#56b5fd")
    table.set_fontsize(14)
    table.scale(1, 3)
    plt.show()

def get_score(y_val, y_pred):
    """
    Get classification score
    :param y_val: validation label
    :param y_pred: predicted label
    :return score: dictionary with scoring
    """
    return {'balanced_accuracy': round(balanced_accuracy_score(y_val, y_pred), 3),
           'precision': round(precision_score(y_val, y_pred), 3),
           'recall': round(recall_score(y_val, y_pred), 3),
           'f1': round(f1_score(y_val, y_pred), 3)}
    
def classification_metrics(y_val, y_pred, title, confusion_matrix=True, precision_recall=True, score=True):
    """
    Plot all classification metrics
    :param y_val: validation label
    :param y_pred: predicted label
    :param confusion_matrix: will show confusion matrix plot if True
    :param precision_recall: will show precision-recall curve if True
    :param score: will show classification metrics if True
    """
    if confusion_matrix:
        plot_conf_matrix(y_val, y_pred, title)
    if score:
        plot_score(y_val, y_pred, title)
    if precision_recall:
        plot_pr_curve(y_val, y_pred, title)


# -

def plot_pie(items, labels, legend, title):
    """
    param items: items to be displayed
    param labels: labels for items to be displayed
    param title: Figure title text 
    """
    plt.figure(figsize=(6,6))
    plt.pie(x=items,
            labels=labels,
            autopct='%1.2f%%',
            textprops={'fontsize':14},
            explode = [0.01 for i in items],
            colors=sns.color_palette('Set2'))
    plt.title(label=title, 
              fontdict={"fontsize":16},
              pad=20)
    plt.legend(legend, loc="upper right")
    plt.show()


# -----
# ## Cleaning functions

# +
def drop_constant_cols(df: pd.DataFrame):
    """
    Drop constant columns from given dataframe
    (constants are not relevant for prediction).
    :param df: dataframe object
    """
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)
            
def drop_unique_cols(df: pd.DataFrame):
    """
    Drop columns that contain only unique values
    (Unique IDs are not relevant for prediction).
    :param df: dataframe object
    """
    for col in df.columns:
        if df[col].nunique() == len(df):
            df.drop(col, axis=1, inplace=True) 

def drop_temp_employments(df: pd.DataFrame, lang='en'):
    """
    Drop temporary employments 
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        idxs_to_drop = df[(df["employment_form"] == "Midlertidig")].index
    else:
        idxs_to_drop = df[(df["Ansettelsesform (A-mel.)"] == "Midlertidig")].index
    df.drop(idxs_to_drop , inplace=True)
    print(f'{len(idxs_to_drop)} entries removed - temporary employees')

def drop_on_leave_without_salary(df: pd.DataFrame, lang='en'):
    """
    Remove people on leave without salary
    """
    if lang == 'en':
        idxs_to_drop = df[(df["employment_type_code"] == "Permisjon u/lønn")].index
    else:
        idxs_to_drop = df[(df["Ansattform"] == "Permisjon u/lønn")].index
    df.drop(idxs_to_drop , inplace=True)
    print(f'{len(idxs_to_drop)} entries removed - people on leave without salary')
    
def drop_addresses(df: pd.DataFrame, lang='en'):
    """
    Drop address columns as they are not relevant for prediction:
    1) "Adresse 1"
    2) "Adresse 2"
    3) "Adresse 3"
    4) "Postnummer ved lønnskjøring"
    5) "Poststed ved lønnskjøring" columns 
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df.drop(["address_1", 
                 "address_2", 
                 "address_3", 
                 "post_code", 
                 "post_office"], axis=1, inplace=True)
    else:
        df.drop(["Adresse 1", 
                 "Adresse 2", 
                 "Adresse 3", 
                 "Postnummer ved lønnskjøring", 
                 "Poststed ved lønnskjøring"], axis=1, inplace=True)   

def drop_duplicate_info_cols(df: pd.DataFrame, lang='en'):
    """
    There are columns that are 1 to 1 with other columns:
    1) 'Sluttårsak kode' and 'Sluttårsak kode navn'
    2) 'Sluttårsak (A-mel.)' and 'Sluttårsak kode'
    3) 'Stillingsgruppe kode' and 'Stillingsgruppe navn'
    4) 'Stillingskode' and 'Stillingskode tekst'
    5) 'Stillingsbetegnelse' and 'Stillingskode'
    6) 'Lokasjon Post nr' and 'Lokasjon Post navn'
    7) 'Lokasjon Kommune nr' and 'Lokasjon Kommune navn'
    8) 'Ansattform' and 'Ansattform tekst'
    9) 'Årsakskode kode' and 'Årsakskode beskrivelse'
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df.drop(['termination_code_desc', 
                 'position_group_name', 
                 'position_name', 
                 'org_unit_post_office', 
                 'org_unit_municipality_name',
                 'employment_type_name',
                 'reason_for_change_text',
                 'position_description',
                 'termination_reason'], axis=1, inplace=True)
    else:
        df.drop(['Sluttårsak kode navn', 
                 'Stillingsgruppe navn', 
                 'Stillingskode tekst', 
                 'Lokasjon Post navn', 
                 'Lokasjon Kommune navn',
                 'Ansattform tekst',
                 'Årsakskode beskrivelse',
                 'Stillingsbetegnelse',
                 'Sluttårsak (A-mel.)'], axis=1, inplace=True)

def drop_non_ordinary_employments(df: pd.DataFrame, lang='en'):
    """
    Keep only 'Ordinært' (Normal) form of employments in the dataset.
    Drop all rows with column 'Arbeidsforhold type (A-mel.)' not equal to 'Ordinært'
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        idxs_to_drop = df[(df['employment_type'] != 'Ordinært')].index
    else:
        idxs_to_drop = df[(df['Arbeidsforhold type (A-mel.)'] != 'Ordinært')].index
    df.drop(idxs_to_drop , inplace=True)
    print(f'{len(idxs_to_drop)} entries removed - non ordinary employees')


def drop_other_termination_codes(df: pd.DataFrame, lang='en'):
    """
    Drop entries with termination code different than 99 (99 is for termination at will)
    """
    if lang == 'en':
        idxs_to_drop = df[(df['termination_code']!='99') & (df['termination_code'].notnull())].index
    else:
        idxs_to_drop = df[(df['Sluttårsak kode']!='99') & (df['Sluttårsak kode'].notnull())].index
    df.drop(idxs_to_drop , inplace=True)
    print(f'{len(idxs_to_drop)} entries removed - due to different termination code than 99 (ended at will)')

    
def drop_datetime_cols(df: pd.DataFrame):
    """
    Drop datetime columns from dataset
    :param df: dataframe object
    """
    datetime_columns = list(df.select_dtypes(include="datetime").columns.values)
    df.drop(columns=datetime_columns, inplace=True)
    
    
def drop_irrelevant_data(df: pd.DataFrame, lang='en'):
    """
    Drop irrelevant columns (constants, unique IDs, duplicate info data, etc.)
    :param df: dataframe object
    :param lang: language of the columns
    """
    before = len(df)
    drop_datetime_cols(df)
    drop_constant_cols(df)
    # drop_unique_cols(df)
    drop_on_leave_without_salary(df, lang)
    # drop_other_termination_codes(df, lang)
    drop_addresses(df, lang)
    drop_duplicate_info_cols(df, lang)
    drop_non_ordinary_employments(df, lang)
    drop_temp_employments(df, lang)
    
    if lang == 'en':
        # drop Antall - column used for exports (not relevant for us)
        df.drop(['export_code'], axis=1, inplace=True)
        # drop Status - there are many cases when employee is terminated but status is active - irrelevant data
        df.drop(['status'], axis=1, inplace=True)
        # drop Gjelder fra and Gjelder til columns - irrelevant dates
        df.drop(['valid_from', 'valid_to'], axis=1, inplace=True)
        # drop 'Ansettelsesform (A-mel.)' - irrelevant and almost none deviating data (also duplicated to other cols)
        df.drop(['employment_form'], axis=1, inplace=True)
        # drop 'Timer per uke' column - irrelevant because of almost none deviation in data
        df.drop(['hours_per_week'], axis=1, inplace=True)
        # drop "Ansattandel" - employed_percentage - not relevant
        df.drop(['employed_percentage'], axis=1, inplace=True)
        # drop "Utbetalingsprosent" - payout_percentage - not relevant
        df.drop(['payout_percentage'], axis=1, inplace=True)
        # drop "Tilstedeprosent" - attendance_pecentage - not relevant
        df.drop(['attendance_pecentage'], axis=1, inplace=True)
        # drop 'Ansattform' - employment_type_code - not relevant
        df.drop(['employment_type_code'], axis=1, inplace=True)
        
    else:
        df.drop(['Antall'], axis=1, inplace=True)
        df.drop(['Status'], axis=1, inplace=True)
        df.drop(['Gjelder fra', 'Gjelder til'], axis=1, inplace=True)
        df.drop(['Ansettelsesform (A-mel.)'], axis=1, inplace=True)
        df.drop(['Timer per uke'], axis=1, inplace=True)
        df.drop(['Ansattandel'], axis=1, inplace=True)
        df.drop(['Utbetalingsprosent'], axis=1, inplace=True)
        df.drop(['Tilstedeprosent'], axis=1, inplace=True)
        df.drop(['Ansattform'], axis=1, inplace=True)
    
    print(f"Dropped columns in total: {before - len(df)}")


# -

# ## Uniform data types:

def uniform_dtypes(df: pd.DataFrame, lang='en'):
    """
    :param df: dataframe object
    """
    
    if lang == 'en':
        # Uniform datetime columns:
        dt_cols = ['export_date', 'birth_date', 'first_employeed_date', 'start_date', 'end_date_employment', 'stop_date_person']
        for col in dt_cols:
            df[col] = pd.to_datetime(df[col])

        # Uniform categorical columns:
        cat_cols= []

        # Uniform numerical columns:
        num_cols = []

        # Uniform boolean columns:
        bool_cols = []
    else:
        # Uniform datetime columns:
        dt_cols = ['År-Mnd', 'Fødselsdato', 'Første gang ansatt dato', 'Startdato', 'Sluttdato', 'Sluttdato person']
        for col in dt_cols:
            df[col] = pd.to_datetime(df[col])

        # Uniform categorical columns:
        cat_cols= []

        # Uniform numerical columns:
        num_cols = []

        # Uniform boolean columns:
        bool_cols = []


# -----
# ## Fill missing data functions

# +
def fill_age(df: pd.DataFrame, lang='en'):
    """
    Calculate age based on the birth date (Fødselsdato) 
    and date when export was made (År-Mnd)
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df['birth_date'] = pd.to_datetime(df['birth_date'])
        df['export_date'] = pd.to_datetime(df['export_date'])
        df['age'] = ((df['export_date'] - df['birth_date']).dt.days)/365
        # round by transformation to integer:
        df['age'] = df['age'].apply(lambda x: int(x))
    else:
        df['Fødselsdato'] = pd.to_datetime(df['Fødselsdato'])
        df['År-Mnd'] = pd.to_datetime(df['År-Mnd'])
        df['Alder'] = ((df['År-Mnd'] - df['Fødselsdato']).dt.days)/365
        # round by transformation to integer:
        df['Alder'] = df['Alder'].apply(lambda x: int(x))

def fill_civil_status(df: pd.DataFrame, lang='en'):
    """
    Fill missing civil status columns and set its value to Unknown (Ukjent)
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df["civil_status"].fillna('Ukjent', inplace=True)
    else:
        df["Sivilstatus"].fillna('Ukjent', inplace=True)
    
def fill_unit_mun_number(df: pd.DataFrame, lang='en'):
    """
    Fill missing Org. unit municipality number
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df['org_unit_municipality'] = pd.to_numeric(df['org_unit_municipality'], errors='coerce').fillna(-9999).astype(int)
    else:
        df['Lokasjon Kommune nr'] = pd.to_numeric(df['Lokasjon Kommune nr'], errors='coerce').fillna(-9999).astype(int)

def fill_termination_reason_code(df: pd.DataFrame, lang='en'):
    """
    Fill termination reason code ('Sluttårsak kode') for resigned employees only
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df['termination_code'].mask(df['end_date_employment'].notnull() & df['termination_code'].isnull(), -9999, inplace=True)
    else:
        df['Sluttårsak kode'].mask(df['Sluttdato'].notnull() & df['Sluttårsak kode'].isnull(), -9999, inplace=True)

def fix_position_codes_to_numeric(df: pd.DataFrame):
    """
    Fix position codes manually, because with transformation it will change its original values
    """
    df.loc[df["position_group_code"] == "UKJENT", "position_group_code"] = 9999
    df.loc[df["position_code"] == "UKJENT", "position_code"] = 9999
    df['position_group_code'] = df['position_group_code'].astype(int)
    df['position_code'] = df['position_code'].astype(int)    
    
def fill_missing_data(df: pd.DataFrame, lang='en'):
    """
    Fill missing data: 
    * Age
    * Civil status
    * Unit municipality number
    * Termination reason code
    :param df: dataframe object
    :param lang: language of the columns
    """
    fill_age(df, lang)
    fill_unit_mun_number(df, lang)
    fill_civil_status(df, lang)
    fill_termination_reason_code(df, lang)
    fix_position_codes_to_numeric(df)


# -

# -----
# ## Helpers

# +
def get_missing_data(dfdf: pd.DataFrame):
    """
    Print out missing data in given dataframe
    :param df: dataframe object
    """
    print(df.isna().sum())
    

def __months_diff(a, b):
    """
    substract two dates and returns number of months as integer
    """
    return abs((a - b)/np.timedelta64(1, 'M'))


def __eval_label(a, b, treshold):
    """
    Evaluate label
    """
    if a>0:
        if b<=treshold:
            return True
    return False


# -

# ## Grouping methods

# +

def __group_by_salary(salary):
    """
    Salary categorisation
    :param salary: employee salary
    """
    salary = float(salary.replace(",", ".") if type(salary) == str else salary)
    if salary < 400000:
        return 3  # "< 400000"
    elif salary < 500000:
        return 4  # "400000-500000"
    elif salary < 600000:
        return 5  # "500000-600000"
    elif salary < 700000:
        return 6  # "600000-700000"
    elif salary < 800000:
        return 7  # "700000-800000"
    elif salary < 1000000:
        return 8  # "800000-1000000" 
    elif salary < 1500000:
        return 9  # "1000000-1500000"
    else:
        return 10  # ">1500000"

def __group_by_age(age):
    """
    Age categorisation
    :param age: employee age as integer
    """
    start = int(age/10) * 10
    end = (int(age/10)+1) * 10 - 1
    # return f"{start}-{end}"
    return start


def create_salary_groups(df: pd.DataFrame, lang='en'):
    """
    Create salary groups
    :param df: dataframe object
    :param lang: language of the columns
    """
    df["salary_group"] = df.apply(lambda x: __group_by_salary(x["salary"]), axis=1) if lang=='en' else df.apply(lambda x: __group_by_salary(x["Årslønn"]), axis=1)
    
    
def create_age_groups(df: pd.DataFrame, lang='en'):
    """
    Create age groups
    :param df: dataframe object
    :param lang: language of the columns
    """
    df["age_group"] = df.apply(lambda x: __group_by_age(x["age"]), axis=1) if lang=='en' else df.apply(lambda x: __group_by_age(x["Alder"]), axis=1)
    

def aggregate_salaries(df: pd.DataFrame, lang='en'):
    """
    Aggregate salary based on export year
    :param df: dataframe object
    :param lang: language of the columns
    """
    df['salary'] = df['salary'].str.replace(',', '').astype(float)
    df["avg_pos_salary"] = df.groupby(['position_code', 'export_year']).salary.transform('mean').astype(int)
    df["med_position_salary"] = df.groupby(['position_code', 'export_year']).salary.transform('median').astype(int)
    df["avg_pos_grp_salary"] = df.groupby(['position_group_code', 'export_year']).salary.transform('mean').astype(int)
    df["med_pos_grp_salary"] = df.groupby(['position_group_code', 'export_year']).salary.transform('median').astype(int)
    df["avg_age_grp_salary"] = df.groupby(['age_group', 'export_year']).salary.transform('mean').astype(int)
    df["med_age_grp_salary"] = df.groupby(['age_group', 'export_year']).salary.transform('median').astype(int)
    df["avg_sex_salary"] = df.groupby(['sex', 'export_year']).salary.transform('mean').astype(int)
    df["med_sex_salary"] = df.groupby(['sex', 'export_year']).salary.transform('median').astype(int)

    
def aggregate_age(df: pd.DataFrame, lang='en'):
    """
    Aggregate age in groups
    :param df: dataframe object
    :param lang: language of the columns
    """
    df["avg_age_on_pos"] = df.groupby(['position_code', 'export_year']).age.transform('mean').astype(int)
    # df["avg_age_pos_grp"] = df.groupby(['position_group_code', 'export_year']).age.transform('mean').astype(int)
    
def calculate_time_spent_on_position(df: pd.DataFrame, lang='en'):
    """
    Calculate average months spent by employees on the specific position.
    :param df: dataframe object
    :param lang: language of the columns
    """
    df['avg_job_dur'] = df.groupby(['position_code']).months_active.transform('mean').astype(int)
    

def calculate_movement_score(df: pd.DataFrame, lang='en'):
    """
    Calculate movement score - how many times has employee changed position
    :param df: dataframe object
    :param lang: language of the columns
    """
    df['movement_score'] = df.groupby(['employment_id']).position_code.transform(lambda x: len(set(x)))
    
    
def create_groups(df: pd.DataFrame, lang='en'):
    """
    Create groups
    :param df: dataframe object
    :param lang: language of the columns
    """
    create_salary_groups(df)
    create_age_groups(df)
    aggregate_salaries(df)
    aggregate_age(df)
    calculate_time_spent_on_position(df)
    calculate_movement_score(df)


# -

# ## Features engineering 

# +
# employed somewhere else at the same time

def create_entry_year_month(df: pd.DataFrame, lang='en'):
    """
    Each entry has a registration year and month information ('År-Mnd')
    We need to keep this info for grouping purposes and train/test splits.
    Thus we will create new numeric cols keeping year and monmth separately as type integer.
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df['export_year'] = df['export_date'].astype(str).apply(lambda x: x.split('-')[0]).astype(int)
        df['export_month'] = df['export_date'].astype(str).apply(lambda x: x.split('-')[1]).astype(int)
    else:
        df['export_year'] = df['År-Mnd'].astype(str).apply(lambda x: x.split('-')[0]).astype(int)
        df['export_month'] = df['År-Mnd'].astype(str).apply(lambda x: x.split('-')[1]).astype(int)

def create_birth_year(df: pd.DataFrame, lang='en'):
    """
    Create birth year column.
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df["birth_year"] = df["birth_date"].apply(lambda x: x.date().year).astype(int)
    else:
        df["birth_year"] = df["Fødselsdato"].apply(lambda x: x.date().year).astype(int)
    
def create_active_months(df: pd.DataFrame, lang='en'):
    """
    Calculate number of months that person is active in company
    :param df: dataframe object
    :param lang: language of the columns
    """
    if lang == 'en':
        df['months_active'] = abs((df['first_employeed_date']- df['export_date'])/np.timedelta64(1, 'M'))
    else:
        df['months_active'] = abs((df['Første gang ansatt dato']- df['År-Mnd'])/np.timedelta64(1, 'M'))
    df['months_active'] = df['months_active'].astype(int)

    
def create_is_multi_emp(df: pd.DataFrame, lang='en'):
    """
    Check if person have multiple active employments at the same time
    :param df: dataframe object
    :param lang: language of the columns
    """
    pass


def create_resigned_in_months(df: pd.DataFrame, lang='en'):
    """
    Calculate number of months from start of employment to its end
    :param df: dataframe object
    :param lang: language of the columns
    """
    df['resigned_after_x_months'] = abs((df['end_date_employment'] - df['start_date'])/np.timedelta64(1, 'M')).fillna(-9999, inplace=True).astype(int)

    
def create_new_features(df: pd.DataFrame, lang='en'):
    """
    Apply features engineering methods.
    :param df: dataframe object
    :param lang: language of the columns
    """
    create_entry_year_month(df)
    create_birth_year(df)
    create_active_months(df)
    # create_resigned_in_months(df)


# -

# ## Transformation

# +
def transform_boolean_cols(df: pd.DataFrame):
    """
    Function that will transform boolean columns to integer.
    :param df: dataframe object
    """
    boolean_columns = list(df.select_dtypes(include="bool").columns.values)
    df[boolean_columns] = df[boolean_columns].astype(int)
    
    
def transform_categoric_cols(df: pd.DataFrame):
    """
    Function that will transform categorical columns into the numeric values
    LabelEncoder - Encode target labels with value between 0 and n_classes-1.
    :param df: dataframe object
    """
    categorical_columns = list(df.select_dtypes(include="object").columns.values)
    enc = LabelEncoder()

    for column in categorical_columns:
        df[column] = enc.fit_transform(df[column].astype(str))


# -

# ## Labels

def create_labels(df: pd.DataFrame):
    """
    Create labels
    :param df: dataframe object
    """
    df['resigned_after_months'] = df.apply(lambda x: __months_diff(x['end_date_employment'], x['start_date']), axis=1).fillna(-9999).astype(int)
    df['remaining_months'] = df.apply(lambda x: __months_diff(x['end_date_employment'], x['export_date']), axis=1).fillna(-9999).astype(int)
    df[learning_setup['label_column']] = df.apply(lambda x: __eval_label(x['resigned_after_months'], x['remaining_months'], MONTHS), axis=1)


# --------
# ## Preprocessing

# +
# load data from csv:
df = pd.read_csv(learning_setup['data_path'], sep=";", low_memory=False)

# rename columns to english names:
df = df.rename(nor_to_eng_names, axis=1)
original_df = df.copy(deep=True)
# -

print(f"Original data length: {len(df)}")

# Group cols based on type:
boolean_columns = list(df.select_dtypes(include="bool").columns.values)
categorical_columns = list(df.select_dtypes(include="object").columns.values)
datetime_columns = list(df.select_dtypes(include="datetime").columns.values)
numerical_columns = list(df.select_dtypes(include="number").columns.values)

uniform_dtypes(df)

fill_missing_data(df)

create_new_features(df)

create_groups(df)

create_labels(df)

# transform dataframe from cumulative to the narrow:
df = df.sort_values('export_date').groupby('person_id').tail(1)

# remove cases with salary less than 1000:
before = len(df)
df = df.drop(df[df['salary']<1000].index)
print(f"Dropped columns: {before - len(df)}")

df['employment_type'].value_counts()

drop_irrelevant_data(df)

df['employment_type'].value_counts()

transform_boolean_cols(df)

transform_categoric_cols(df)

df

get_missing_data(df)

# Drop columns with missing data (if any)
before = len(df)
df = df.dropna()
print(f"Dropped columns: {before-len(df)}")

df.info()

# -------
# # Trend prediction
# ## Fluctuation over time - Is there any trend?

resigned_employees = df[df["resigned_in_6m"] == 1]
fluct = resigned_employees.groupby(['export_year', 'export_month']).size()
fluct.plot(color='green', marker='o', linestyle='dashed',
     linewidth=1, markersize=3, figsize=(18,6), title='Fluctuation over time - all employees', fontsize=9, xticks=range(0, len(fluct), 6))

# ----
# ## Why do employees leave?

why_leave['termination_reason'].value_counts()

why_leave = original_df.copy(deep=True)

why_leave = why_leave.sort_values('export_date').groupby('person_id').tail(1)

# +
items = why_leave['termination_code_desc'].value_counts()

# Calculate value counts
value_counts = why_leave['termination_code_desc'].value_counts()


# Create a figure and axes
plt.figure(figsize=(10, 6))

# Create a bar plot using value_counts index as x-values and value_counts values as y-values
bars = plt.bar(value_counts.index, value_counts.values, color=plt.cm.Paired.colors)

# Create legend items using the bars' colors
legend_items = [plt.Line2D([0], [0], marker='o', color=color, label=f'{item} - {count}') for item, count, color in zip(value_counts.index, value_counts.values, plt.cm.Paired.colors)]

# Add legend with custom legend items
plt.legend(handles=legend_items, title='Legend')

plt.xlabel("Categories")
plt.ylabel("Count")
plt.title("Termination reason distribution", fontsize=20, pad=15)

plt.show()
# -

#
# ## Which position groups have the highest fluctuation?

resigned_employees['position_group_code'].value_counts().head(3)

items = resigned_employees['position_group_code'].value_counts(normalize=True)
threshold = 0.02
items = items.where(items >= threshold, other='0')
plot_pie(items=items,
        labels=items.keys(),
        legend=items.keys(),
        title='Most fluctuated position groups')

original_df[original_df['position_group_code'] == '6'][['position_name', 'position_group_code']].head(2)

original_df[original_df['position_group_code'] == '8'][['position_name', 'position_group_code']].head(2)

original_df[original_df['position_group_code'] == '5'][['position_name', 'position_group_code']].head(2)

original_df[original_df['position_group_code'] == '9'][['position_name', 'position_group_code']].head(2)

# ## Plot yearly fluctuation of all employees that ended at will

grouped = resigned_employees.groupby(['export_year', 'export_month']).size().reset_index(name='count')
years = resigned_employees['export_year'].unique()  
for year in years:
    filtered_grouped = grouped[grouped['export_year'] == year]
    reveal_missing = [i for i in filtered_grouped['export_month'].unique()]
    for i in range(1,13):
        if i not in reveal_missing:
            reveal_missing.append(i)
            filtered_grouped.loc[year+i] = [year, i, 0]
    filtered_grouped = filtered_grouped.sort_values(['export_year', 'export_month'])
    plt.figure(figsize=(12, 2))
    plt.plot(filtered_grouped['export_month'], filtered_grouped['count'], linestyle='dashed', marker='o', markersize=4)
    plt.xlabel('month')
    plt.ylabel('resigned employees')
    plt.title(f'Fluctuation (all employees ended at will) {year}')
    plt.xticks(filtered_grouped['export_month'])
    plt.show()

# ## Plot most fluctuated group of employees (Senior rådgiver)

resigned_grp_6 = resigned_employees[resigned_employees['position_group_code']==6]

# +
grouped = resigned_grp_6.groupby(['export_year', 'export_month']).size().reset_index(name='count')
years = resigned_grp_6['export_year'].unique()

for year in years:
    filtered_grouped = grouped[grouped['export_year'] == year]
    reveal_missing = [i for i in filtered_grouped['export_month'].unique()]
    for i in range(1,13):
        if i not in reveal_missing:
            reveal_missing.append(i)
            filtered_grouped.loc[year+i] = [year, i, 0]
    filtered_grouped = filtered_grouped.sort_values(['export_year', 'export_month'])
    plt.figure(figsize=(12, 2))
    plt.plot(filtered_grouped['export_month'], filtered_grouped['count'], linestyle='dashed', marker='o', markersize=4)
    plt.xlabel('month')
    plt.ylabel('resigned employees')
    plt.title(f'Fluctuation (position group 6) {year}')
    plt.xticks(filtered_grouped['export_month'])
    plt.show()


# -

# ----------
# # Data split

# +
def split_pos_neg(df):
    """
    Split dataset to positive and negative class based on number of days that defines label
    :param df: data as dataframe
    :return: positive and negative classes as dataframes
    :rtype: tuple
    """
    label = learning_setup.get('label_column')
    
    # Positive class:
    df_positive = df.loc[(df[label] == 1)]

    # Negative class:
    df_negative = pd.concat([df, df_positive, df_positive]).drop_duplicates(keep=False)
    
    return df_positive, df_negative


def split_test_train(df):
    """
    Split data to test and train set according to the test size and number of days to test on
    defined in learning setup
    :return: test and train dataframes
    :rtype: tuple
    """
    # Separate test set by randomly picking test data from dataset
    # Split according to the test size defined in learning setup
    df_test = df.sample(frac=learning_setup.get('test_size'))

    # Exclude test data from training:
    df = pd.concat([df, df_test, df_test]).drop_duplicates(keep=False)
    
    return df_test, df


# -

# Test/train split:
df_test, df = split_test_train(df=df)

# +
train_cols = ['sex',
             'age',
             'nationality',
             'civil_status',
             'org_unit',
             'position_group_code',
             'position_code',
             'movement_score',
             'months_active',
             'salary',
             'salary_group',
             'avg_pos_salary',
             'avg_age_on_pos',
             'resigned_in_6m']

df = df[train_cols]
df_test = df_test[train_cols]
# -

df_test

df_test[df_test['resigned_in_6m']==1].tail()

df_test[df_test['resigned_in_6m']==0].tail()

a = df_test['resigned_in_6m'].value_counts().to_dict()
plot_pie(items=df_test['resigned_in_6m'].value_counts(), 
         labels=['Active', 'Resigned'],
         legend=[f'Active ({a.get(0)})', f'Resigned ({a.get(1)})'],
         title="Employees distribution - Test data")

# Positibve/negative split:
pos, neg = split_pos_neg(df)

# +
# Separate label_column from the rest of the columns
label_column = learning_setup.get('label_column')
x = df.loc[:, df.columns != label_column]
y = df[label_column].copy()

# Split data:
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=learning_setup.get('validation_size'))

# Check the split dataset shapes:
x_train.shape, x_val.shape, y_train.shape, y_val.shape
# -

# ----
# ----
# # PCA - Principal Component Analysis

# +
# Scale the dataset
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_val_std = sc.transform(x_val)
pca = PCA()
x_train_pca = pca.fit_transform(x_train_std)
exp_var_pca = pca.explained_variance_ratio_

# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

# Plot
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# -

x_train.columns

exp_var_pca

# -------
# ------
# # Models

# **classification goal** - will employee quit in next 6 months?
#
# **regression goal** - how many employees on the same position will quit in next 6 months  
#
# - use as less as possible feaures for training

# ----
# ----
# # Classification
# * 1) Decision Tree
# * 2) Random Forest
# * 3) XgBoost

# ------
# -----
# ## 1) Decision tree

# +
# create and fit model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)

# Make predictions
y_pred_bin = dt_model.predict(x_val)
y_pred_c = dt_model.predict_proba(x_val)

# Save validation scoring:
validation_scoring = get_score(y_val, y_pred_bin)

# Model evaluation:
classification_metrics(y_val, y_pred_bin, 'Decision tree')
# -

# ## 1.1) Decsison tree - model testing

# +
# Create positive and negative classes for test set
df_test_pos, df_test_neg = split_pos_neg(df=df_test)

# Merge positive and negative classes into test set:
df_test = pd.concat([df_test_pos, df_test_neg])

# Drop columns with missing data:
before = len(df_test)
df_test = df_test.dropna()
print(f"Dropped columns: {before-len(df_test)}")

# Separate label_column from the rest of the columns
x_test = df_test.loc[:, df_test.columns != label_column]
y_test = df_test[label_column].copy()

# Make predictions on test data:
y_test_pred_bin = dt_model.predict(x_test)

# Save test scoring:
test_scoring = get_score(y_test, y_test_pred_bin)

# Evaluation:
classification_metrics(y_test, y_test_pred_bin, 'Decision tree - testing')
# -

# ---
# ---
# ## 2) Random forest

# +
# Create and fit the model:
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

# Make predictions
y_pred_bin = rf_model.predict(x_val)
y_pred_c = rf_model.predict_proba(x_val)

# Save validation scoring:
validation_scoring = get_score(y_val, y_pred_bin)

# Model evaluation:
classification_metrics(y_val, y_pred_bin, 'Random forest')
# -

# ## 2.1) Random forest - hyperparameters tuning 

'''
param_dist = {'n_estimators': [50, 100, 150, 200, 250, 300],
              'max_depth': [5,6,7,8,9,10,11,12,13,14,15]}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)

# Fit the random search object to the data
rand_search.fit(x_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)  

max_depth = rand_search.best_params_.get('max_depth')
n_estimators = rand_search.best_params_.get('n_estimators')
'''

# ## 2.2) Random forest - model testing

# +
# Create positive and negative classes for test set
df_test_pos, df_test_neg = split_pos_neg(df=df_test)

# Merge positive and negative classes into test set:
df_test = pd.concat([df_test_pos, df_test_neg])

# Drop columns with missing data:
before = len(df_test)
df_test = df_test.dropna()
print(f"Dropped columns: {before-len(df_test)}")

# Separate label_column from the rest of the columns
x_test = df_test.loc[:, df_test.columns != label_column]
y_test = df_test[label_column].copy()

# Make predictions on test data:
y_test_pred_bin = rf_model.predict(x_test)

# Save test scoring:
test_scoring = get_score(y_test, y_test_pred_bin)

# Evaluation:
classification_metrics(y_test, y_test_pred_bin, 'Random forest - testing')
# -

# -------
# -------
# ## 3) XGBoost

# +
# Apply weighting if data_weighting parameter is True:

if learning_setup.get('data_weighting'):
    counter = y_train.value_counts()
    train_neg_count = counter[0]
    train_pos_count = counter[1]
    scale_pos_weight = (train_neg_count/train_pos_count)
else:
    scale_pos_weight = 1

learning_rate = 0.1
max_depth = 20
min_child_weight = 3
gamma = 0.3
colsample_bytree = 0.5    

# XGboost configuration:
conf = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'colsample_bytree': colsample_bytree,
    'gamma': gamma,
    'gpu_id': -1,
    'importance_type': 'gain',
    'interaction_constraints': '',
    'learning_rate': learning_rate,
    'max_delta_step': 0,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,
    'missing': 1,
    'monotone_constraints': '()',
    'n_estimators': 100,
    'n_jobs': 12,
    'num_parallel_tree': 1,
    'random_state': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'scale_pos_weight': scale_pos_weight,
    'subsample': 1,
    'tree_method': 'exact',
    'validate_parameters': 1,
    'verbosity': None,
}

# Create model:
xgb_model = XGBClassifier(**conf)

# Fit model:
xgb_model.fit(
    x_train,
    np.squeeze(y_train),
    eval_set=[(x_val, y_val),],
    eval_metric="auc",
    verbose=False,
)

# Make predictions
y_pred_bin = xgb_model.predict(x_val)
y_pred_c = xgb_model.predict_proba(x_val)

# Save validation scoring:
validation_scoring = get_score(y_val, y_pred_bin)

# Model evaluation:
classification_metrics(y_val, y_pred_bin, 'XGboost')
# -

# ## 3.1) XGBoost - Model testing

# +
# Create positive and negative classes for test set
df_test_pos, df_test_neg = split_pos_neg(df=df_test)

# Merge positive and negative classes into test set:
df_test = pd.concat([df_test_pos, df_test_neg])

# Drop columns with missing data:
before = len(df_test)
df_test = df_test.dropna()
print(f"Dropped columns: {before-len(df_test)}")

# Separate label_column from the rest of the columns
x_test = df_test.loc[:, df_test.columns != label_column]
y_test = df_test[label_column].copy()

# Make predictions on test data:
y_test_pred_bin = xgb_model.predict(x_test)

# Save test scoring:
test_scoring = get_score(y_test, y_test_pred_bin)

# Evaluation:
classification_metrics(y_test, y_test_pred_bin, 'XGboost - testing')
# -

# -----
# -----
# # Experiment - removing employees older than 50 years from testing data

df_test_u50 = df_test[df_test['age'] <50]

df_test_u50

a = df_test_u50['resigned_in_6m'].value_counts().to_dict()
plot_pie(items=df_test_u50['resigned_in_6m'].value_counts(), 
         labels=['Active', 'Resigned'],
         legend=[f'Active ({a.get(0)})', f'Resigned ({a.get(1)})'],
         title="Employees distribution - Test data age under 50")

# +
# Separate label_column from the rest of the columns
x_test = df_test_u50.loc[:, df_test_u50.columns != label_column]
y_test = df_test_u50[label_column].copy()

# Make predictions on test data:
y_test_pred_bin = xgb_model.predict(x_test)

# Save test scoring:
test_scoring = get_score(y_test, y_test_pred_bin)

# Evaluation:
classification_metrics(y_test, y_test_pred_bin, 'XGBoost - testing on u50')
# -




