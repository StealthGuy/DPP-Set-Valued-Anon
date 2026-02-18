from src.models.partition import Partition

class PartitionAnonymizer:
    def __init__(self, k, root_node, total_domain_size):
        self.k = k
        self.root_node = root_node
        self.total_domain_size = total_domain_size
        self.final_partitions = []

    def anonymize(self, partition):
        # Se non ci sono più nodi da espandere (o sono tutti foglie o tutti esclusi)
        node_to_expand = self._pick_node(partition)
        
        if not node_to_expand:
            self.final_partitions.append(partition)
            return

        # Distribuzione dei dati
        buckets = self._distribute_data(partition, node_to_expand)
        valid_sub_partitions = self._balance_partitions(buckets, partition, node_to_expand)

        if not valid_sub_partitions:
            # creiamo una NUOVA partizione con il nodo escluso aggiunto.
            # evita accumulo non controllato di excluded_nodes su oggetti condivisi tra chiamate ricorsive.
            new_excluded = set(partition.excluded_nodes)
            new_excluded.add(node_to_expand)

            # ripristiniamo la rappresentazione delle transazioni al cut corrente
            for t in partition.transactions:
                t.update_representation(partition.hierarchy_cut)

            new_partition = Partition(
                partition.transactions,
                list(partition.hierarchy_cut),
                new_excluded
            )
            self.anonymize(new_partition)
            return

        # ricorsione sui nuovi bucket validi
        for sub_p in valid_sub_partitions:
            self.anonymize(sub_p)

    def _pick_node(self, partition):
        best_node = None
        max_gain = -1
        
        for node in partition.hierarchy_cut:
            # Scegliamo solo nodi non foglie e non precedentemente falliti
            if not node.is_leaf() and node not in partition.excluded_nodes:
                gain = self._calculate_info_gain(partition, node)
                if gain > max_gain:
                    max_gain = gain
                    best_node = node
        return best_node

    def _calculate_info_gain(self, partition, node):
        """
        Il guadagno è calcolato pesando la differenza di NCP per ognitransazione che contiene almeno un item coperto dal nodo.
        """
        before_ncp = node.get_ncp(self.total_domain_size)
        after_ncp_per_child = {child: child.get_ncp(self.total_domain_size) for child in node.children}

        total_gain = 0.0
        for t in partition.transactions:
            for original_item in t.original_items:
                # risaliamo per vedere se questo item è coperto dal nodo candidato
                curr = original_item
                while curr is not None:
                    if curr == node:
                        # L'item è coperto da questo nodo -> calcoliamo il guadagno cercando quale figlio lo coprirebbe dopo l'espansione
                        curr2 = original_item
                        covering_child = None
                        while curr2 is not None:
                            if curr2 in after_ncp_per_child:
                                covering_child = curr2
                                break
                            curr2 = curr2.parent
                        
                        if covering_child is not None:
                            gain = before_ncp - after_ncp_per_child[covering_child]
                            total_gain += gain
                        break
                    curr = curr.parent

        return total_gain

    def _distribute_data(self, partition, node_to_expand):
        """
        Ogni bucket riceve una COPIA indipendente del new_cut per evitare che oggetti condivisi vengano corrotti da modifiche successive.
        """
        base_cut = [n for n in partition.hierarchy_cut if n != node_to_expand]
        base_cut.extend(node_to_expand.children)

        buckets = {}
        for t in partition.transactions:
            # ogni chiamata usa il base_cut, ma ogni bucket memorizza una copia propria per sicurezza
            t.update_representation(base_cut)
            signature = tuple(sorted([n.name for n in t.current_representation]))
            if signature not in buckets:
                buckets[signature] = ([], list(base_cut))  # copia
            buckets[signature][0].append(t)

        return list(buckets.values())

    def _balance_partitions(self, buckets, parent_partition, attempted_node):
        valid_partitions = []
        leftover_transactions = []

        for b_transactions, b_cut in buckets:
            if len(b_transactions) >= self.k:
                valid_partitions.append(Partition(b_transactions, b_cut))
            else:
                leftover_transactions.extend(b_transactions)

        if leftover_transactions:
            # Tentativo di recupero dei leftover rubando da partizioni valide 
            while len(leftover_transactions) < self.k and valid_partitions:
                source_p = valid_partitions.pop()
                leftover_transactions.extend(source_p.transactions)

            if len(leftover_transactions) >= self.k:
                # Creiamo il leftover con il cut originale del genitore escludendo il nodo che ha appena fallito 
                new_excluded = set(parent_partition.excluded_nodes)
                new_excluded.add(attempted_node)
                # Aggiorna la rappresentazione delle transazioni residue al cut del genitore
                for t in leftover_transactions:
                    t.update_representation(parent_partition.hierarchy_cut)
                    
                valid_partitions.append(Partition(leftover_transactions, parent_partition.hierarchy_cut, new_excluded))
            else:
                # Se non riusciamo a formare un gruppo di k lo split fallisce
                return []

        return valid_partitions