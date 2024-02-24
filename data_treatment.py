def merge_datasets(filenames, output_filename):
    """
    Merge datasets from a list of filenames.
    """
    dataset = []
    for filename in filenames:
        with open(filename, "r") as file:
            dataset += file.read().splitlines()
    
    with open(output_filename, "w") as file:
        for line in dataset:
            file.write(line + "\n")


def split_string_based_on_size(string, size, complete_character = "_"):
    """
    Split a string into a list of strings based on a given size.
    If the offset_string lenght is not equeal to size, complete it with complete_character.
    """
    offset_size = len(string) % size

    if offset_size != 0:
        offset_string = complete_character * (size - offset_size)
        string += offset_string

    return [string[i:i+size] for i in range(0, len(string), size)]


def transfom_dataset_to_split_dataset(dataset, size, complete_character = "_"):
    """
    Transform a dataset into a split dataset based on a given size.
    If the offset_string lenght is not equeal to size, complete it with complete_character.
    """
    return [split_string_based_on_size(string, size, complete_character) for string in dataset]


def get_dataset_from_file(file_path):
    """
    Get a dataset from a file.
    """
    with open(file_path, "r") as file:
        return file.read().splitlines()
    

def get_new_dataset_from_file(file_path, size, complete_character = "_"):
    """
    Get a new dataset from a file.
    """
    word_dataset = transfom_dataset_to_split_dataset(get_dataset_from_file(file_path), size, complete_character)

    association_vetor = []
    for i in (word_dataset):
        for j in i:
            association_vetor.append(i)

    # make word dataset 1d
    word_dataset = [j for i in word_dataset for j in i]

    return word_dataset, association_vetor
