class HierarchyNode:
    """Rappresenta un nodo nell'albero di generalizzazione"""
    
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self._leaves_count = None  # Foglie totali sotto questo nodo 

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_leaves_count(self, total_domain_size=None):
        """
        Calcola quante foglie sono coperte da questo nodo. Utile per NCP
        """
        if self.is_leaf():
            return 1
        
        if self._leaves_count is None:
            self._leaves_count = sum(child.get_leaves_count() for child in self.children)
        return self._leaves_count

    def get_ncp(self, total_items_in_domain):
        """
        Calcola il Normalized Certainty Penalty per questo nodo (NCP)
        """
        if self.is_leaf():
            return 0.0
        return self.get_leaves_count() / total_items_in_domain
    
    @staticmethod
    def from_dict(data, parent=None):
        """
        Metodo ricorsivo per creare un albero HierarchyNode da un dizionario JSON.
        """
        node = HierarchyNode(data["name"], parent=parent)
        
        # Se il nodo ha dei figli, li creiamo ricorsivamente
        if "children" in data:
            for child_data in data["children"]:
                child_node = HierarchyNode.from_dict(child_data, parent=node)
                node.add_child(child_node)
        
        return node