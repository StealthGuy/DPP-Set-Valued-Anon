class Partition:
    def __init__(self, transactions, hierarchy_cut, excluded_nodes=None):
        self.transactions = transactions
        self.hierarchy_cut = hierarchy_cut
        # nodi che abbiamo già provato a espandere in questa partizione e hanno fallito
        self.excluded_nodes = excluded_nodes if excluded_nodes else set()

    def __len__(self):
        return len(self.transactions)