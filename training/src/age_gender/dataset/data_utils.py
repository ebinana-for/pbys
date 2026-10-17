"""Utility functions fr age/gender evaluation and processing."""

import numpy as np

fairface_age_categories = [
    "0;2",
    "3;9",
    "10;19",
    "20;29",
    "30;39",
    "40;49",
    "50;59",
    "60;69",
    "70;120",
]

forvia_age_categories = ["0;14", "15;24", "25;64", "64;120"]


def cat2age(age_cat):
    age_low, age_high = age_cat.split(";")
    return np.median([int(age_low), int(age_high)])


def age2cat(age, age_categories):
    for age_cat in age_categories:
        age_low, age_high = age_cat.split(";")
        if age >= int(age_low) and age <= int(age_high):
            return age_cat
