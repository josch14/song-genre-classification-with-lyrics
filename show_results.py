
from asyncio import constants
import os
import json

# local imports
from constants import *


def build_dict():
    results_dict = {}

    for model in MODELS:
        model_dict = {}
        for n_target_genres in range(2, 12+1):
            genre_dict = {}
            for measure in MEASURES:
                genre_dict[measure] = 0.00
            model_dict[n_target_genres] = genre_dict
        results_dict[model] = model_dict

    with open(os.path.join(EVALUATION_FOLDER, f"results_overview.json"), 'w') as file:
        json.dump(results_dict, file)

if __name__ == '__main__':
    # build_dict()