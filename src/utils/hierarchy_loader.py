import json
from src.models.item_hierarchy import HierarchyNode

class HierarchyLoader:
    @staticmethod
    def load_from_json(filepath):
        """Carica il file JSON e restituisce la radice dell'albero."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        root = HierarchyNode.from_dict(data)
        
        # estraiamo anche la lista "piatta" di tutte le foglie (per il generatore di transazioni)
        leaves = []
        def find_leaves(node):
            if node.is_leaf():
                leaves.append(node)
            for child in node.children:
                find_leaves(child)
        
        find_leaves(root)
        return root, leaves
    
# FUNZIONE USATA SOLO NEI TEST per verificare la costruzione di gerarchia, quando non presente, come nel dataset del paper
def build_balanced_hierarchy_for_fanout(fan_out, n_items=200):
    """Costruisce gerarchia bilanciata artificiale con fan_out dato."""
    from src.models.item_hierarchy import HierarchyNode
    leaves = [HierarchyNode(f"item_{i}") for i in range(n_items)]
    current = list(leaves)
    depth = 0
    while len(current) > 1:
        depth += 1
        next_layer = []
        for i in range(0, len(current), fan_out):
            group = current[i: i + fan_out]
            parent = HierarchyNode(f"L{depth}_N{len(next_layer)}")
            for child in group:
                parent.add_child(child)
            next_layer.append(parent)
        current = next_layer
    root = current[0]
    root.name = "ALL"
    return root, leaves