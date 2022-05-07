import numpy as np
import os
from constants import MODELS, RESULTS_FOLDER

if __name__ == '__main__':

    for model_name in MODELS:
        print(f"\nModel: {model_name}")
        statistics = []
        for n_target_genres in range(2, 12+1):
            # load results
            save_path = os.path.join(RESULTS_FOLDER, f"{model_name}_{n_target_genres}.npy")
            history=np.load(save_path, allow_pickle=True).item()

            # get statistics
            val_accuracies = history['val_accuracy']
            best_accuracy = max(val_accuracies)

            model_statistics = [n_target_genres, round(best_accuracy*100, 2), val_accuracies.index(best_accuracy)]
            statistics.append(model_statistics) # genres, best accuracy, epoch
            print(f"{model_statistics[0]} categories: {model_statistics[1]} (epoch {model_statistics[2]})")