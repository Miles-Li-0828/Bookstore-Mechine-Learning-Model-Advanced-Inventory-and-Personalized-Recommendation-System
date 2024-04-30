import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pycountry
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def weighted_age_dict(user_file):
    users_df_copy = pd.read_csv(user_file)
    users_df_copy.dropna(subset=["User-Age"], inplace=True)

    users_df_copy["User-Age"] = pd.to_numeric(
        users_df_copy["User-Age"], errors="coerce"
    )
    users_df_copy.dropna(subset=["User-Age"], inplace=True)
    users_df_copy = users_df_copy.loc[users_df_copy["User-Age"] < 100]
    users_df_copy = users_df_copy.loc[users_df_copy["User-Age"] > 0]

    # Define the bins of
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # lable all the bins
    labels = [
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
    ]

    users_df_copy["Age-Group"] = pd.cut(
        users_df_copy["User-Age"], bins=bins, labels=labels
    )
    users_df_copy = users_df_copy.sort_values(by="User-Age", ascending=True)

    # Count the number of each age group
    age_group_counts = users_df_copy["Age-Group"].value_counts()

    # Calculate the age-group percentage
    age_group_percentages = (age_group_counts / len(users_df_copy)) * 100
    age_group_percentages_dict = age_group_percentages.to_dict()
    return age_group_percentages_dict


def ages_imputation(age_group_percentages_dict, users_df):
    """Handle the missing data of users' ages"""
    random_filled_ages = list(age_group_percentages_dict.keys())
    total_percentage = sum(age_group_percentages_dict.values())
    weights = [v / total_percentage for v in age_group_percentages_dict.values()]

    # Remove non-digit characters from 'User-Age' column
    users_df["User-Age"] = users_df["User-Age"].str.replace(r"\D", "", regex=True)

    # Convert 'User-Age' column to numeric
    users_df["User-Age"] = pd.to_numeric(users_df["User-Age"], errors="coerce")

    # Count the rows where user age is null
    count_null_age = users_df["User-Age"].isnull().sum()

    # Count the rows where user age is less than 10 and greater than 99
    count_extreme_ages = (
        (users_df["User-Age"] < 10) | (users_df["User-Age"] > 99)
    ).sum()

    # Total number of rows with errored age or null
    error_age_count = count_null_age + count_extreme_ages

    # Generate a list of random numbers between 10 and 99
    random.seed(42)
    random_ages = []
    for i in range(error_age_count):
        random_ages.append(np.random.choice(random_filled_ages, p=weights))

    # Create a mask for rows where user age is null or falls outside the range [10, 99]
    error_age_rows = (
        users_df["User-Age"].isnull()
        | (users_df["User-Age"] < 10)
        | (users_df["User-Age"] > 99)
    )

    # Extract the indexes of the rows that satisfy the condition
    error_age_indexes = users_df.index[error_age_rows].tolist()

    fill_age = {idx: ages for idx, ages in zip(error_age_indexes, random_ages)}

    # Iterate over the fill_age dictionary and update the DataFrame
    for idx, age in fill_age.items():
        users_df.loc[idx, "User-Age"] = age
    return users_df


def country_imputation(users_df):
    # List all the countries in the data
    country_list = list(users_df["User-Country"].dropna().unique())

    # List row indexes where User-Country is "n/a"
    na_indexes = list(users_df[users_df["User-Country"].str.strip() == 'n/a"'].index)

    # List row indexes where User-Country is null
    nan_indexes = list(
        users_df["User-Country"][users_df["User-Country"].isnull()].index
    )

    # Count total number of rows with null
    count_nan = len(nan_indexes) + len(na_indexes)

    # Generate random countries for all null rows
    random.seed(42)
    random_countries = [random.choice(country_list) for _ in range(count_nan)]

    fill_country = {
        idx: countries
        for idx, countries in zip(nan_indexes + na_indexes, random_countries)
    }

    # Iterate over the fill_states dictionary and update the DataFrame
    for idx, country in fill_country.items():
        users_df.loc[idx, "User-Country"] = country
    return users_df


def city_imputation(users_df):
    # List all the cities in the data
    city_list = list(users_df["User-City"].dropna().unique())

    # List row indexes where User-City is null
    nan_indexes = list(users_df["User-City"][users_df["User-City"].isnull()].index)

    # Count total number of rows with null
    count_nan = len(nan_indexes)

    # Generate random cities for all null rows
    random.seed(42)
    random_cities = [random.choice(city_list) for _ in range(count_nan)]

    fill_city = {idx: cities for idx, cities in zip(nan_indexes, random_cities)}

    # Fill gaps in User-City column
    users_df["User-City"] = users_df["User-City"].fillna(fill_city)
    return users_df


def state_imputation(users_df):
    # List all the states in the data
    state_list = list(users_df["User-State"].dropna().unique())

    # List row indexes where User-State is "n/a"
    na_indexes = users_df[
        (users_df["User-State"].str.strip() == "n/a")
        | (users_df["User-State"].str.strip() == "")
    ].index

    # Count total number of rows with null
    count_na = len(na_indexes)

    # Generate random states for all null rows
    random.seed(42)
    random_states = [random.choice(state_list) for _ in range(count_na)]

    fill_states = {idx: states for idx, states in zip(na_indexes, random_states)}

    # Iterate over the fill_states dictionary and update the DataFrame
    for idx, state in fill_states.items():
        users_df.loc[idx, "User-State"] = state
    return users_df


def author_imputation(df):
    df["Book-Author"].fillna("NO AUTHOR", inplace=True)
    return df


def discretising(users_df, rating_df, books_df):
    # Define the bins of
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # lable all the bins
    labels = [
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100",
    ]

    # Discretise the data
    users_df["Age-Group"] = pd.cut(users_df["User-Age"], bins=bins, labels=labels)
    users_df = users_df.sort_values(by="User-Age", ascending=True)

    # Make a histogram to show the distribution of all discretised data
    plt.figure(figsize=(10, 6))
    plt.hist(users_df["Age-Group"], bins=len(labels), edgecolor="black", alpha=0.7)
    plt.title("Distribution of Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("DistributionOfAgeGroups.png")

    merged_df = pd.merge(users_df, rating_df, on="User-ID", how="inner")
    merged_df = pd.merge(merged_df, books_df, on="ISBN", how="inner")

    # Discretise the data for rating into low, medium and high
    merged_df["Book-Rating"] = merged_df["Book-Rating"].astype(int)
    rate_bins = [0, 4, 7, 11]
    rate_labels = ["low", "medium", "high"]
    merged_df["Rating_Category"] = pd.cut(
        merged_df["Book-Rating"], bins=rate_bins, labels=rate_labels, right=False
    )

    return merged_df


def text_process(merged_df):
    countries = merged_df["User-Country"]
    countries_upper = countries.str.upper()
    country_mapping = {
        "U.S.A.": "UNITED STATES",
        "U.S.A": "UNITED STATES",
        "USA": "UNITED STATES",
        "U.S. OF A.": "UNITED STATES",
        "U.S.A>": "UNITED STATES",
        "U.S>": "UNITED STATES",
        "AMERICA": "UNITED STATES",
        "UNITED STATE": "UNITED STATES",
        "UNITED STATES OF AMERICA": "UNITED STATES",
        "ENGLAND": "UNITED KINGDOM",
        "U.K.": "UNITED KINGDOM",
        "UNITED KINGDOMN": "UNITED KINGDOM",
        "WALES": "UNITED KINGDOM",
        "SCOTLAND": "UNITED KINGDOM",
        "GREAT BRITAIN": "UNITED KINGDOM",
    }

    countries_cleaned = np.char.strip(
        np.char.replace(countries_upper.values.astype(str), '"', "")
    )
    countries_cleaned = np.array(
        [country_mapping.get(country, country) for country in countries_cleaned]
    )

    # filter out words that are not country, randomly replace them with countries that are already existed
    known_countries = set(country.name.upper() for country in pycountry.countries)

    # Preprocess countries_cleaned to remove non-country values
    valid_countries_cleaned = [
        country for country in countries_cleaned if country in known_countries
    ]

    # Replace non-country values with a random choice from valid_countries_cleaned
    filtered_countries = [
        (
            country
            if country in known_countries
            else random.choice(valid_countries_cleaned)
        )
        for country in countries_cleaned
    ]

    merged_df["User-Country"] = filtered_countries

    str_cols = [
        "User-City",
        "User-State",
        "User-Country",
        "Book-Title",
        "Book-Author",
        "Book-Publisher",
    ]
    merged_df[str_cols] = merged_df[str_cols].apply(lambda x: x.str.lower())
    return merged_df


def compute_probability(col):
    """
    Compute the probability of a certain event
    """
    return col.value_counts() / col.shape[0]


def compute_entropy(col):
    """
    Compute the entropy of a certain event
    """
    entropy = 0
    probabilities = compute_probability(col)
    for prob in probabilities:
        if prob == 0:
            term = 0
        else:
            term = -prob * np.log2(prob)
        entropy += term
    return entropy


def compute_conditional_entropy(x, y):
    """
    Compute the conditional entropy between two random variables.
    Specifically, the conditional entropy of Y given X.
    """
    probability_x = compute_probability(x)

    temp_df = pd.DataFrame({"X": x, "Y": y})

    conditional_entropy = 0

    # for unique event x_i
    for x_i in x.unique():
        # get the data for Y given X=x_i
        y_given_x = temp_df.loc[temp_df["X"] == x_i, "Y"]

        # compute the conditional entropy
        conditional_entropy += probability_x[x_i] * compute_entropy(y_given_x)

    return conditional_entropy


def compute_information_gain(x, y):
    """
    Compute the information gain between an attribute and class label
    """
    return compute_entropy(y) - compute_conditional_entropy(x, y)
