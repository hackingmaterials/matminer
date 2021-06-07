from matminer.datasets import get_available_datasets, get_dataset_columns, \
    get_dataset_column_description, get_dataset_reference, \
    get_dataset_num_entries, get_dataset_description, get_dataset_citations


# Functions used for auto-generating dataset description pages.

__authors__ = 'Daniel Dopp <dbdopp@lbl.gov>'


def make_rst_doc_title(string):
    return "\n".join(["=" * len(string), string, "=" * len(string)])


def make_rst_subtitle(string):
    return "\n".join(["-" * len(string), string, "-" * len(string)])


def make_rst_subsection(string):
    return "\n".join([string, "-" * len(string)])


if __name__ == "__main__":
    # make a big table of datasets

    datasets = get_available_datasets(print_format=None)


    print(make_rst_doc_title("Table of Datasets"))
    print(f"Find a table of all {len(datasets)} datasets available in matminer here.")


    print("\n.. list-table::")
    print("   :align: left")
    print("   :widths: 20 70 10")
    # print("   :width: 70%")
    print("   :header-rows: 1\n")
    print("   * - Name")
    print("     - Description")
    print("     - Entries")

    for dataset in datasets:
        info = get_dataset_description(dataset)
        sentences = info.split(". ")
        sentence1 = sentences[0]
        n_entries = get_dataset_num_entries(dataset)
        print(f"   * - :code:`{dataset}`")
        print(f"     - {sentence1}")
        print(f"     - {n_entries}")

    print(make_rst_doc_title("Dataset info"))

    # For each dataset give the name, description,
    # num_entries, columns, and reference string
    for dataset in get_available_datasets(print_format=None):
        # Name, description, and number of entries output
        print(make_rst_subtitle(dataset))
        print(get_dataset_description(dataset))
        print("\n**Number of entries:** {}\n".format(
            get_dataset_num_entries(dataset)
        ))

        # Get all columns and find the max length for border string
        dataset_columns = get_dataset_columns(dataset)
        colname_max_length = max(map(len, dataset_columns))

        # Get all column descriptions and find max length for border string
        column_descriptions = [get_dataset_column_description(dataset, column)
                               for column in dataset_columns]
        desc_max_length = max(map(len, column_descriptions))

        print(".. list-table::")
        print("   :align: left")
        print("   :widths: 20 80")
        print("   :header-rows: 1\n")
        print("   * - Column")
        print("     - Description")
        for c in dataset_columns:
            print(f"   * - :code:`{c}`")
            column_description = get_dataset_column_description(dataset, c)
            print(f"     - {column_description}")
        print("\n\n")

        # Give dataset reference
        print("**Reference**\n")
        print(get_dataset_reference(dataset))
        print("\n\n")

        # Give bibtex citations
        print("**Bibtex Formatted Citations**\n")

        print(".. code-block:: text")
        print("\n")
        for citation in get_dataset_citations(dataset):
            print("\t" + citation.replace("\n"," "))
            print()
        print("\n\n")
