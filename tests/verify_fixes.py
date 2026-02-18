
import sys
import os

# WRITTEN BY GEMINI AI

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.models.item_hierarchy import HierarchyNode
from src.models.transaction import Transaction
from src.models.partition import Partition
from src.core.anonymizer import PartitionAnonymizer
from src.utils.metrics import calculate_database_ncp # Import metric for testing

def reproduce_bug():
    print("=== TEST 1: Dirty Representation Bug ===")
    # 1. Create Hierarchy
    # Root -> A -> (A1, A2)
    root = HierarchyNode("Root")
    node_a = HierarchyNode("A", parent=root)
    root.add_child(node_a)
    
    leaf_a1 = HierarchyNode("A1", parent=node_a)
    node_a.add_child(leaf_a1)
    
    leaf_a2 = HierarchyNode("A2", parent=node_a)
    node_a.add_child(leaf_a2)
    
    # 2. Create Transactions
    t1 = Transaction(1, [leaf_a1])
    t1.current_representation = [root] 
    
    t2 = Transaction(2, [leaf_a2])
    t2.current_representation = [root]

    transactions = [t1, t2]
    
    # 3. Create Initial Partition
    initial_cut = [root]
    partition = Partition(transactions, initial_cut)
    
    # 4. Run Anonymizer (k=2 forces failure of split A -> A1, A2)
    k = 2
    domain_size = 2 # A1, A2
    anonymizer = PartitionAnonymizer(k, root, total_domain_size=domain_size)
    anonymizer.anonymize(partition)
    
    # 5. Check Results
    if not anonymizer.final_partitions:
        print("[!] No final partitions found!")
        return

    final_p = anonymizer.final_partitions[0]
    cut_names = [n.name for n in final_p.hierarchy_cut]
    print(f"Partition Cut: {cut_names}")
    
    bug_found = False
    for t in final_p.transactions:
        rep_names = [n.name for n in t.current_representation]
        print(f"TID {t.tid}: {rep_names}")
        
        # Check if representation matches cut
        if rep_names != cut_names:
            bug_found = True
            
    if bug_found:
        print("\n[!] BUG PERSISTS: Transactions representation does not match partition cut.")
    else:
        print("\n[*] SUCCESS: Representations correctly match partition cut.")

def verify_metric():
    print("\n=== TEST 2: NCP Metric Calculation ===")
    # Root -> A -> (A1, A2)
    root = HierarchyNode("Root")
    node_a = HierarchyNode("A", parent=root)
    root.add_child(node_a)
    leaf_a1 = HierarchyNode("A1", parent=node_a)
    node_a.add_child(leaf_a1)
    leaf_a2 = HierarchyNode("A2", parent=node_a)
    node_a.add_child(leaf_a2)
    
    # Transaction: {A1, A2} -> Generalized to {A}
    # A covers 2 leaves (A1, A2). Domain Size = 10 (arbitrary for calculation)
    # NCP(A) = leaves_in_A / domain_size = 2 / 10 = 0.2
    
    t = Transaction(1, [leaf_a1, leaf_a2])
    t.current_representation = [node_a] # Generalized state
    
    total_domain_size = 10
    
    # Expected NCP:
    # Item A1 covered by A -> NCP 0.2
    # Item A2 covered by A -> NCP 0.2
    # Transaction NCP (sum) = 0.4
    # Database NCP (avg) = 0.4 / 2 items = 0.2
    
    # Old Incorrect Logic:
    # Set(rep) = {A}. 
    # Sum NCP({A}) = 0.2
    # Avg = 0.2 / 2 = 0.1 (Undershoots)
    
    score = calculate_database_ncp([t], total_domain_size)
    expected = 0.2
    
    print(f"Calculated NCP: {score}")
    print(f"Expected NCP:   {expected}")
    
    if abs(score - expected) < 0.0001:
        print("[*] SUCCESS: Metric calculated correctly per item.")
    else:
        print("[!] FAILURE: Metric calculation mismatch.")

if __name__ == "__main__":
    reproduce_bug()
    verify_metric()
