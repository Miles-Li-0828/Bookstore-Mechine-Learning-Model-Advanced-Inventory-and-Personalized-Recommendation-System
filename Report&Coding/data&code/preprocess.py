import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


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


def imputation(age_group_percentages_dict, users_df):
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

    """Handle the missing data of countries"""
    # List all the countries in the data
    country_list = list(users_df["User-Country"].dropna().unique())

    # List row indexes where User-Country is null
    nan_indexes = list(
        users_df["User-Country"][users_df["User-Country"].isnull()].index
    )

    # Count total number of rows with null
    count_nan = len(nan_indexes)

    # Generate random countries for all null rows
    random.seed(42)
    random_countries = [random.choice(country_list) for _ in range(count_nan)]

    fill_country = {
        idx: countries for idx, countries in zip(nan_indexes, random_countries)
    }

    # Fill gaps in User-Country column
    users_df["User-Country"] = users_df["User-Country"].fillna(fill_country)
    return users_df


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

    # Merge user infomations and rating informations
    merged_df = pd.merge(users_df, rating_df, on="User-ID", how="inner")
    merged_df = pd.merge(merged_df, books_df, on="ISBN", how="inner")

    return merged_df


def text_process(merged_df):
    return
