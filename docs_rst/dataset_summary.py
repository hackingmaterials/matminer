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
    # Print out page title header and description of the page
    print(make_rst_doc_title("Datasets Available in Matminer"))
    print("\nBelow you will find descriptions and reference data on each "
          "available dataset, ordered by load_dataset() keyword argument\n\n")

    # For each dataset give the name, description,
    # num_entries, columns, and reference string
    for dataset in get_available_datasets(print_datasets=False):
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

        # Give column info table header
        name_header = "Column"
        desc_header = "Description"
        colname_border_length = max(colname_max_length, len(name_header))
        desc_border_length = max(desc_max_length, len(desc_header))
        print("=" * colname_border_length, "=" * desc_border_length)
        print(name_header + " " * (colname_border_length - len(name_header)),
              desc_header)
        print("=" * colname_border_length, "=" * desc_border_length)

        # Give table rows
        for column, description in zip(dataset_columns, column_descriptions):
            print(column + " " * (colname_border_length - len(column)),
                  description)
        print("=" * colname_border_length, "=" * desc_border_length)
        print("\n\n")

        # Give dataset reference
        print("**Reference**\n")
        print(get_dataset_reference(dataset))
        print("\n\n")

        # Give bibtex citations
        print("**Bibtex Formatted Citations**\n")
        for citation in get_dataset_citations(dataset):
            print(citation)
            print()
        print("\n\n")
