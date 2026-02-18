import random
import csv
from typing import List, Dict, Set
from src.models.transaction import Transaction
from src.models.item_hierarchy import HierarchyNode

class TransactionGenerator:
    """
    Generatore di transazioni per set-valued data.
    Modella il comportamento tramite bias di categoria e correlazione.
    """

    def __init__(self, root_node: HierarchyNode):
        self.root = root_node
        self.leaves = self._collect_leaves(root_node)
        self.categories = root_node.children
        # Mappa ogni categoria alle sue foglie per generare cluster realistici
        self.category_map = {cat: self._collect_leaves(cat) for cat in self.categories}

    def _collect_leaves(self, node: HierarchyNode) -> List[HierarchyNode]:
        """Recupera ricorsivamente tutte le foglie (item reali) sotto un nodo"""
        if node.is_leaf():
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def generate_transactions( self, count: int, avg_size: int = 4, category_bias: float = 0.7, correlation_strength: float = 0.5 ) -> List[Transaction]:
        dataset = []
        # Determiniamo il limite fisico
        max_possible_items = len(self.leaves) 

        for i in range(count):
            main_cat = random.choice(self.categories)
            preferred_items = self.category_map[main_cat]
            
            # Calcoliamo la dimensione desiderata
            size = max(1, int(random.gauss(avg_size, 1.5)))
            
            # Impediamo il loop infinito limitando size al dominio reale
            size = min(size, max_possible_items) 
            
            current_items: Set[HierarchyNode] = set()
            
            while len(current_items) < size:
                if random.random() < category_bias and preferred_items:
                    item = random.choice(preferred_items)
                else:
                    item = random.choice(self.leaves)
                
                current_items.add(item)
                
                if random.random() < correlation_strength and len(current_items) < size:
                    siblings = [c for c in item.parent.children if c != item and c not in current_items]
                    if siblings and len(current_items) < size: 
                        current_items.add(random.choice(siblings))

            dataset.append(Transaction(i, list(current_items)))
        
        return dataset

    def export_to_csv(self, transactions: List[Transaction], filepath: str):
        """Esporta il dataset nel formato ID, Items come nel paper"""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['TID', 'Items'])
            for t in transactions:
                items_str = ", ".join([it.name for it in t.original_items])
                writer.writerow([t.tid, items_str])
        print(f"[*] Export completato: {len(transactions)} transazioni in {filepath}")