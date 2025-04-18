from test_train_sets import *
import numpy as np
from evaluate import evaluate

test_set = get_test_set()

overall_scores = []
excitment_scores = []

for i in test_set:
    overall_scores.append(get_score(i, "Overall"))
    excitment_scores.append(get_score(i, "Excited"))

overall_numpy = np.array(overall_scores)
excited_numpy = np.array(excitment_scores)

mean_overall = overall_numpy.mean()
mean_excited = excited_numpy.mean()

mean_overall_guesses = np.ones(len(overall_scores)) * mean_overall
mean_excited_guesses = np.ones(len(excitment_scores)) * mean_excited

print(mean_overall_guesses)
print(mean_excited_guesses)

evaluate(overall_numpy, mean_overall_guesses)
# Mean Absolute Relative Error: 0.0775
# Pearson: nan (p-value: nan)

evaluate(excited_numpy, mean_excited_guesses)
# Mean Absolute Relative Error: 0.1064
# Pearson: nan (p-value: nan)
