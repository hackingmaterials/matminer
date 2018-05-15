



class BaseDataRetrieval:

    def get_dataframe(self, criteria, prpoerties):
        raise NotImplementedError("get_dataframe() is not defined!")

    def api_link(self):
        raise NotImplementedError("api_link() is not defined!")