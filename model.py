import subprocess
import re

import numpy as np
import pandas as pd

from data_treatment import get_new_dataset_from_file


def compute_scores(alphabet_file:str, train_file: str, test_file: str, r=4, n=10):
    return re.sub(
        "[b']",
        "",
        str(
            subprocess.check_output(
                f"java -jar -alphabet {alphabet_file} negsel2.jar -self {train_file} -n {n} -r {r} -c -l < {test_file}",
                shell=True
            )
        )
    ).split('\\n')[:-1]


def apply_model(alphaber_file_path, train_word_dataset, test_word_dataset, n, r):
    """
    Apply the model.
    """
    # create a file with the train word dataset
    with open("train_word_dataset.txt", "w") as file:
        for word in train_word_dataset:
            file.write(word + "\n")
    
    # create a file with the word dataset
    with open("test_word_dataset.txt", "w") as file:
        for word in test_word_dataset:
            file.write(word + "\n")


    # compute scores
    result = compute_scores(alphaber_file_path, "train_word_dataset.txt", "test_word_dataset.txt", r, n)

    return result


def get_anomally_score(result, association_vetor, label_file, aggregation_function = "mean"):
    """
    Get the anomally score from the result based on a aggregation function.
    """
    if aggregation_function not in ["mean"]:
        raise ValueError("aggregation_function must be 'mean'")

    # open label file
    with open(label_file, "r") as file:
        label = file.read().splitlines()

    # send label to a pandas dataframe
    label = pd.DataFrame(label, columns = ["label"])
    label["anomaly_score"] = 0

    final_result = [[] for i in range(label.shape[0])]
    
    for i in range(len(result)):
        final_result[association_vetor[i]].append(result[i])

    if aggregation_function == "mean":
        for i in range(len(final_result)):
            label["anomaly_score"].iloc[i] = np.mean(final_result[i])
    
    return label


def save_anomally_score(label, output_file):
    """
    Save the anomally score to a file.
    """
    label.to_csv(output_file, index = False)


def main(
    alphaber_file,
    training_data_file,
    word_dataset_file,
    n,
    r,
    label_file,
    output_file,
    complete_character="_"
):
    """
    Main function.
    """
    print("Getting train dataset from file...")
    train_word_dataset, _ = get_new_dataset_from_file(training_data_file, n, complete_character)
    
    print("Getting new test dataset from file...")
    test_word_dataset, association_vetor = get_new_dataset_from_file(word_dataset_file, n, complete_character)
    
    print("Applying model...")
    result = apply_model(alphaber_file, train_word_dataset, test_word_dataset, n, r, complete_character)
    
    print("Getting anomally score for each word...")
    label = get_anomally_score(result, association_vetor, label_file)
    
    print("Saving anomally score to file...")
    save_anomally_score(label, output_file)
    
    print(f"Anomally score saved to file {output_file}")
