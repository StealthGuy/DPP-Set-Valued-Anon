def calculate_database_ncp(transactions, total_domain_size):
    """
    Calcola la perdita di informazione totale del database anonimizzato.
    """
    total_ncp_sum = 0
    total_item_count = 0
    missing_coverage_count = 0

    for t in transactions:
        transaction_ncp = 0
        for original_item in t.original_items:
            covering_node = None
            curr = original_item
            while curr is not None:
                if curr in t.current_representation:
                    covering_node = curr
                    break
                curr = curr.parent

            if covering_node:
                transaction_ncp += covering_node.get_ncp(total_domain_size)
            else:
                # applichiamo penalità massima (NCP=1) per non sottostimare
                missing_coverage_count += 1
                transaction_ncp += 1.0  # penalità massima conservativa

        total_ncp_sum += transaction_ncp
        total_item_count += len(t.original_items)

    if missing_coverage_count > 0:
        print(
            f"[WARN] calculate_database_ncp: {missing_coverage_count} item(s) senza nodo di copertura nel cut corrente. "
            f"Possibile bug nel cut management. Applicata penalità NCP=1.0."
        )

    if total_item_count == 0:
        return 0

    return total_ncp_sum / total_item_count