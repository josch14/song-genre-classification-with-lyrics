import os
import json
import matplotlib.pyplot as plt

# local imports
from constants import *
from lib.utils import *

results_overview_path = os.path.join(EVALUATION_FOLDER, f"results_overview.json")

"""
Methods for producing plots and statistics.
"""
def plot_macro_f1(results_overview: dict):
    pass



"""
Helping Methods.
"""
# read dict with all results
def read_dict():
    with open(results_overview_path, 'r') as file:
        results_overview = json.load(file)
    return results_overview

# fill dict with all results with zeros, and fill it manually
def build_dict():
    results_overview = {}

    for model in MODELS:
        model_dict = {}
        for n_target_genres in range(2, 12+1):
            genre_dict = {}
            for metric in METRIC:
                genre_dict[metric] = 0.00
            model_dict[n_target_genres] = genre_dict
        results_overview[model] = model_dict

    with open(results_overview_path, 'w') as file:
        json.dump(results_overview, file)


# plot metric y=results for all models and x=genres
def plot_metric(min_genre: int, max_genre: int, results_overview: dict, models: list, metric: str, ylim: tuple, y_step: float):
    
    scaling = 0.9
    figure = plt.figure(figsize=(scaling*11, scaling*5), dpi=300)

    genre_list = list(range(min_genre, max_genre+1))
    for model in models:
        values = []
        model_dict = results_overview[model]
        for n_target_genres in genre_list:
            genre_dict = model_dict[str(n_target_genres)]
            values.append(genre_dict[metric])

        plt.plot(genre_list, values, label=MODEL_2_NAME[model])
    
    # x-axis
    plt.xlabel("# genres")
    plt.xticks(np.arange(min_genre, max_genre+1, step=1))

    # y-axis
    plt.ylabel("%")
    plt.yticks(np.arange(ylim[0], ylim[1]+y_step, step=y_step))

    plt.title(METRIC_2_NAME[metric])
    plt.legend()
    plt.ylim(ylim)
    plt.grid()

    figure.savefig(
        os.path.join(EVALUATION_FOLDER, f"{metric}.png"), 
        dpi=300, bbox_inches = 'tight', 
        pad_inches=0.0)


if __name__ == '__main__':
    # build_dict()
    results_overview = read_dict()

    # macro F1
    plot_metric(
        min_genre=3,
        max_genre=12, 
        results_overview=results_overview, 
        models=MODELS, 
        metric=MACRO_F1, 
        ylim=[30.0, 80.00],
        y_step=5.0)