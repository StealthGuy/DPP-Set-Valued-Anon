import os
from src.models.transaction import Transaction
from src.models.item_hierarchy import HierarchyNode


class BMSLoader:
    @staticmethod
    def load_transactions(filepath, max_rows=None):
        """
        Legge i file .dat (BMS-WebView, BMS-POS).
        Restituisce: lista di liste di item (ID come stringhe) e set di item univoci.
        """
        raw_transactions = []
        unique_items = set()
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset non trovato: {filepath}")

        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                
                item_ids = line.strip().split()
                if item_ids:
                    raw_transactions.append(item_ids)
                    unique_items.update(item_ids)
                    
        return raw_transactions, sorted(list(unique_items))