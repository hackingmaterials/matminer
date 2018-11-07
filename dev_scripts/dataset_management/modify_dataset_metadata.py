from pprint import pprint
from copy import deepcopy
import json
from itertools import compress
import os


from matminer.datasets.utils import _load_dataset_dict

intro_message = """
How would you like to interact with the dataset metadata?
(1) See Current Datasets
(2) Remove a Dataset
(3) Add a Dataset
(s) Save changes
(q) Quit
"""


def generate_new_dataset(dataset):
    """
    This function collects user input to build a dictionary of dataset metadata
    as well as ensures that metadata is complete and properly formatted. It
    displays the needed metadata fields that have not yet been filled and gives
    the user the option of which ones to fill.

    The current metadata fields are:
    bibtex_refs - a list of strings, each being a bibtex reference
                  to the source(s) of the dataset

    columns - a dictionary mapping string column names to string descriptions
              of each dataset column

    description - string describing the dataset at a high level

    file_type - file type of the dataset being downloaded from cloud storage

    file_hash - SHA 256 cryptographic hash of the dataset file

    num_entries - int listing number of entries in the dataset

    reference - human readable string referencing dataset source

    url - url where dataset is stored in cloud storage

    Args:
        dataset (str): The name of the dataset, more specifically the keyword
        used for lookup, to be placed in the metadata dictionary

    Returns: dict
    """
    bibtex = []
    columns = {}
    description = ""
    file_type = ""
    file_hash = ""
    num_entries = 0
    reference = ""
    url = ""

    name = input("Provide a dataset name (q to cancel): ").strip()
    if name.lower() == 'q':
        return None, None
    while not name or name in dataset:
        name = input("Invalid name, "
                     "provide a dataset name (q to cancel): ").strip()
        if name.lower() == 'q':
            return None, None

    while True:
        current_entry = {
            'bibtex_refs': bibtex,
            'columns': columns,
            'description': description,
            'file_type': file_type,
            'hash': file_hash,
            'num_entries': num_entries,
            'reference': reference,
            'url': url
        }

        needed_attribs = list(compress(
            [key for key in current_entry.keys()],
            map(lambda x: not x, [value for value in current_entry.values()])))

        if needed_attribs:
            print("Still needed elements are: {}".format(
                ", ".join(needed_attribs)
            ))
        else:
            end_choice = input("All attributes filled, "
                               "would you like to finish (Y/n)").strip().lower()
            if end_choice == 'y':
                return name, current_entry

        print("The entry looks like this:")
        pprint(current_entry, indent=4)

        attrib_name = input("What attribute would you "
                            "like to edit? (q to cancel): ")

        if attrib_name.lower() == 'q':
            return None, None

        elif attrib_name == "bibtex_refs":
            finished_adding_entries = False
            while not finished_adding_entries:
                reference_lines = []
                print("Add a bibtex reference, use multiple lines "
                      "for newlines, hit return to finish the entry: ")
                bibtex_ref = input()
                while bibtex_ref:
                    reference_lines.append(bibtex_ref.strip())
                    bibtex_ref = input()
                new_reference = "\n".join(reference_lines).strip()
                if new_reference:
                    print('The following will be added:')
                    print(new_reference)
                    bibtex.append(new_reference)
                    add_another = input("Would you like to add another "
                                        "reference? (Y/n): ").strip().lower()
                    if add_another == "n":
                        finished_adding_entries = True

        elif attrib_name == "columns":
            finished_adding_entries = False
            while not finished_adding_entries:
                column_name = input("Add a column name or edit a column: ")
                while not name:
                    column_name = input("Add a column name: ")
                column_description = input("Add a column description: ")
                while not column_description:
                    column_description = input("Add a column description: ")
                print('The following will be added:')
                print(column_name)
                columns[column_name] = column_description
                print("Current column list:")
                pprint(columns, indent=4)
                add_another = input("Would you like to add another "
                                    "column? (Y/n): ").strip().lower()
                if add_another == "n":
                    finished_adding_entries = True

        elif attrib_name == "description":
            description = input("Add a dataset description: ").strip()

        elif attrib_name == "file_type":
            file_type = input("Add a dataset filetype: ").strip()

        elif attrib_name == "hash":
            file_hash = input("Add a file hash: ").strip()

        elif attrib_name == "num_entries":
            num_entries = int(input("Add the number of entries: ").strip())

        elif attrib_name == "reference":
            reference_lines = []
            print("Add a plain english reference, use multiple lines "
                  "for newlines, hit return to finish the entry: ")
            ref = input()
            while ref:
                reference_lines.append(ref.strip())
                ref = input()
            new_reference = "\n".join(reference_lines).strip()
            if new_reference:
                print('The following will be added:')
                print(new_reference)
                reference = new_reference

        elif attrib_name == "url":
            url = input("Add a file download url: ").strip()

        else:
            print("Invalid option")


if __name__ == '__main__':
    _dataset_dict = _load_dataset_dict()
    with open(".dataset_data_backup.json", 'w') as outfile:
        json.dump(_dataset_dict, outfile, indent=4, sort_keys=True)

    unsaved_changes = False
    _temp_dataset = deepcopy(_dataset_dict)

    quit_flag = False
    while not quit_flag:
        print(intro_message)
        command = input(">>> ")
        command = command.strip().lower()
        # Show current datasets
        if command == "1":
            print("Current Datasets:")
            pprint(_temp_dataset, width=150)
        # Remove a dataset
        elif command == "2":
            print("Current datasets are: {}".format(
                ", ".join([thing for thing in _temp_dataset.keys()])
            ))
            print("What would you like to remove? (hit return to cancel):")
            removal_dataset = input(">>> ")
            if removal_dataset and removal_dataset in _temp_dataset:
                del _temp_dataset[removal_dataset]
                unsaved_changes = True
            elif removal_dataset:
                print("Dataset does not exist")
        # Add a dataset
        elif command == "3":
            dataset_name, dataset_info = generate_new_dataset(_temp_dataset)
            if dataset_name is not None:
                _temp_dataset[dataset_name] = dataset_info
                unsaved_changes = True
        # Save changes
        elif command == "s":
            if not unsaved_changes:
                print("No changes to save")
            else:
                with open(os.path.abspath(os.path.join(
                        os.pardir, os.pardir, "matminer", "datasets",
                        "dataset_metadata.json")), "w") as outfile:
                    json.dump(_temp_dataset, outfile, indent=4, sort_keys=True)
                unsaved_changes = False
        # Quit
        elif command == "q":
            if unsaved_changes:
                print("There are unsaved changes, "
                      "are you sure you want to quit? (Y/n)")
                quit_choice = input(">>> ")
                if quit_choice.strip().lower() == "y":
                    quit_flag = True
            else:
                quit_flag = True
        else:
            print("Invalid command, see below for options:\n")
