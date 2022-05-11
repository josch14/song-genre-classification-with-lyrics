import os
import json
import matplotlib.pyplot as plt
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# local imports
from constants import *
from lib.utils import *

results_overview_path = os.path.join(EVALUATION_FOLDER, f"results_overview.json")


"""
Methods for producing plots and statistics.
"""
# plot metric y=results for all models and x=genres
def plot_metric(
    results_overview: dict,
    min_genre: int, 
    max_genre: int,
    x_tick_names: list,
    models: list, 
    metric: str, 
    ylim: tuple, 
    y_ticks:list):
    
    scaling = 1.0
    figure = plt.figure(figsize=(scaling*11, scaling*4.5), dpi=300)

    genre_list = list(range(min_genre, max_genre+1))
    for model in models:
        values = []
        model_dict = results_overview[model]
        for n_target_genres in genre_list:
            genre_dict = model_dict[str(n_target_genres)]
            values.append(genre_dict[metric])

        plt.plot(genre_list, values, label=MODEL_2_NAME[model], marker=".")
    
    # x-axis
    plt.xlabel("# Genres")
    plt.xticks(np.arange(min_genre, max_genre+1, step=1))
    plt.xticks(genre_list, x_tick_names)

    # y-axis
    plt.ylabel("%")
    plt.yticks(y_ticks)

    plt.title(METRIC_2_NAME[metric])
    plt.legend()
    plt.ylim(ylim)
    plt.grid()

    figure.savefig(
        os.path.join(EVALUATION_FOLDER, f"plot_{metric}.png"), 
        dpi=300, bbox_inches = 'tight', 
        pad_inches=0.0)


# one line are statistic for a pair of (n_genres, model): 
# macro P & R & F1, and weighted P & R & F1
def produce_latex_table_lines(n_genres: int, results_overview: dict):
    print(f"\n----------Table lines for {n_genres} Genres: ----------")
    
    for model in MODELS:
        # build line string
        model_line = MODEL_2_NAME[model] + " & "
        for metric in METRIC:
            value = results_overview[model][str(n_genres)][metric]
            if metric != METRIC[-1]:
                model_line += (str(value) + " & ")
            else:
                model_line += (str(value) + " \\\\")

        print(model_line)



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



if __name__ == '__main__':
    # build_dict()
    results_overview = read_dict()

    """
    LATEX TABLES.
    """
    for n_genres in range(2, 12+1):
        produce_latex_table_lines(n_genres=n_genres, results_overview=results_overview)


    """
    PLOTS.
    """
    min_genre=2
    max_genre=12
    x_tick_names=["2\n[Pop, Rock]", "3\n+Rap", "4\n+Country", "5\n+Reggae", "6\n+Heavy\nMetal", \
            "7\n+Blues", "8\n+Indie", "9\n+Hip Hop", "10\n+Jazz", "11\n+Folk", "12\n+Gospel/\nReligioso"]

    # macro F1
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=MACRO_F1, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])

    # macro Precision
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=MACRO_PRECISION, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])

    # macro Recall
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=MACRO_RECALL, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])

    # weighted F1
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=WEIGHTED_F1, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])

    # weighted Precision
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=WEIGHTED_PRECISION, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])

    # weighted Recall
    plot_metric(
        min_genre=min_genre,
        max_genre=max_genre, 
        x_tick_names=x_tick_names,
        results_overview=results_overview, 
        models=MODELS, 
        metric=WEIGHTED_RECALL, 
        ylim=[43.0, 82.00],
        y_ticks=[45, 50, 55, 60, 65, 70, 75, 80])