import sys
import os
import argparse

# Setup dei path per importare dal modulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.setrecursionlimit(5000) # Aumenta il limite di ricorsione per gerarchie profonde

from src.models.item_hierarchy import HierarchyNode
from src.models.transaction import Transaction
from src.models.partition import Partition
from src.core.anonymizer import PartitionAnonymizer
from src.utils.metrics import calculate_database_ncp
from src.utils.hierarchy_loader import HierarchyLoader
from src.utils.dataset_generator import TransactionGenerator

def setup_paper_scenario():
    """
    Riproduce la gerarchia (Fig. 1) e le transazioni (Tab. 1a) del paper. 
    Usato per printare qualcosa quando non vengono passati parametri al programma.
    """
    root = HierarchyNode("ALL")
    alc = HierarchyNode("Alcohol", parent=root); root.add_child(alc)
    hc = HierarchyNode("Health Care", parent=root); root.add_child(hc)
    
    beer = HierarchyNode("Beer", parent=alc); alc.add_child(beer)
    wine = HierarchyNode("Wine", parent=alc); alc.add_child(wine)
    diaper = HierarchyNode("Diaper", parent=hc); hc.add_child(diaper)
    preg = HierarchyNode("Pregnancy Test", parent=hc); hc.add_child(preg)

    m = {"beer": beer, "wine": wine, "diaper": diaper, "preg": preg}
    transactions = [
        Transaction("Alice", [m["beer"], m["diaper"]]),
        Transaction("Bob",   [m["wine"], m["diaper"], m["preg"]]),
        Transaction("Chris", [m["beer"], m["wine"], m["preg"]]),
        Transaction("Dan",   [m["beer"], m["wine"], m["diaper"], m["preg"]])
    ]
    return transactions, root, 4 

def main():
    parser = argparse.ArgumentParser(description="Anonymization of set-valued data, using He - Naughton algorithm")
    parser.add_argument('--json', type=str, help="Percorso gerarchia JSON")
    parser.add_argument('-k', type=int, default=2, help="Valore k per k-anonymity")
    parser.add_argument('--num', type=int, default=20, help="Numero di transazioni")
    parser.add_argument('--avg_size', type=int, default=4, help="Dimensione media del carrello")
    parser.add_argument('--bias', type=float, default=0.7, help="Bias di categoria (0.0 - 1.0)")
    parser.add_argument('--corr', type=float, default=0.5, help="Forza della correlazione locale (0.0 - 1.0)")
    parser.add_argument('--save_csv', type=str, help="Percorso per salvare il dataset sintetico")
    parser.add_argument('--output', '-o', type=str, help="Percorso per salvare il dataset anonimizzato in CSV")
    args = parser.parse_args()
    
    print("="*55)
    print(" Set-Valued Data Anonymizer: He - Naughton Algorithm")
    print(" DPP Course - Gabriele Alessandria - s5622102")
    print("="*55)

    if args.json:
        if not os.path.exists(args.json):
            print(f"\n[!] ERRORE: File non trovato: {args.json}")
            return
        print(f"[*] Generazione DATI SINTETICI: Gerarchia dati da {args.json}")
        root, all_leaves = HierarchyLoader.load_from_json(args.json)
        
        # inizializzazione del nuovo generatore
        generator = TransactionGenerator(root)
        transactions = generator.generate_transactions(count=args.num, avg_size=args.avg_size, category_bias=args.bias, correlation_strength=args.corr)
        domain_size = len(all_leaves)

        # esportiamo il dataset se richiesto
        if args.save_csv:
            generator.export_to_csv(transactions, args.save_csv)
    else:
        print("[*] Esempio Alice, Bob, Chris, Dan")
        transactions, root, domain_size = setup_paper_scenario()

    # controllo rispetto di k
    num_total = len(transactions)
    if args.k > num_total:
        print(f"\n[!] ERRORE: k ({args.k}) > totale record ({num_total}).")
        return
    
    # esecuzione algoritmo di partizionamento
    initial_partition = Partition(transactions, [root])
    print(f"[*] Avvio anonimizzazione (k={args.k}) su {num_total} record...")
    
    anonymizer = PartitionAnonymizer(k=args.k, root_node=root, total_domain_size=domain_size)
    anonymizer.anonymize(initial_partition)

    # report finale aggregato
    final_list = [t for p in anonymizer.final_partitions for t in p.transactions]
    ncp = calculate_database_ncp(final_list, domain_size)

    print("\n" + " ANALISI DELLE CLASSI DI EQUIVALENZA ".center(55, "="))
    print(f"[*] Totale Partizioni: {len(anonymizer.final_partitions)}")
    print(f"[*] Information Loss (NCP): {ncp:.4f}")
    print("-" * 55)
    print(f"{'GRUPPO':<10} | {'RAPPRESENTAZIONE ANONIMA'}")
    print("-" * 55)

    for p in anonymizer.final_partitions:
        # prendiamo una transazione campione per mostrare la generalizzazione locale
        sample_rep = [node.name for node in p.transactions[0].current_representation]
        print(f"{len(p.transactions):<10} | {sample_rep}")

    print("=" * 55)

    # for p in anonymizer.final_partitions:
    #     avg = sum(len(t.original_items) for t in p.transactions) / len(p.transactions)
    #     print(f"Partizione {len(p.transactions)} tx | avg_items_originali = {avg:.2f}")

    # export dataset anonimizzato se richiesto
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['TID', 'Items'])
            for p in anonymizer.final_partitions:
                for t in p.transactions:
                    rep_str = ", ".join(node.name for node in t.current_representation)
                    writer.writerow([t.tid, rep_str])
    if args.output:
        print(f"[*] Dataset anonimizzato salvato in: {args.output}")

if __name__ == "__main__":
    main()