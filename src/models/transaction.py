class Transaction:
    """Rappresenta id + transazione come descritto nel paper: (id, {item1, item2, ...})"""
    
    def __init__(self, tid, items):
        self.tid = tid
        self.original_items = set(items)  # gli item specifici originali
        # rappresentazione corrente generalizzata degli item
        self.current_representation = []  # NON è un set perchè così ho sempre il numero totale di prodotti nella transazione anche se generalizzati

    def __repr__(self):
        return f"TID {self.tid}: {self.current_representation}"
    
    def update_representation(self, new_cut):
        """
        Aggiorna la rappresentazione basandosi sui nodi del nuovo Hierarchy Cut.
        Per ogni item originale, trova quale nodo del cut lo copre.
        """
        new_rep = set()
        for leaf in self.original_items:
            # risaliamo la gerarchia della foglia finche non troviamo un nodo nel cut
            curr = leaf
            while curr is not None:
                if curr in new_cut:
                    new_rep.add(curr)
                    break
                curr = curr.parent
        self.current_representation = list(new_rep)