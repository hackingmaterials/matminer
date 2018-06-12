from sklearn.base import TransformerMixin, BaseEstimator


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    A utility for extracting a column from a DataFrame in a sklearn pipeline,
    for example in a FeatureUnion pipeline to featurize a dataset.

    Helper class for making sklearn pipelines with matminer.

    See (http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html)

    Args:
        label : The label of the column to select.
    """
    def __init__(self, label):
        self.label = label

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.label]


class DropExcluded(BaseEstimator, TransformerMixin):
    """
    Transformer for removing unwanted columns from a dataframe.
    Passes back the remaining columns.

    Helper class for making sklearn pipelines with matminer.

    Args:
        excluded (list of labels): A list of column labels to drop from the dataframe
    """

    def __init__(self, excluded):
        self.excluded = excluded

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df.drop(self.excluded, axis=1)