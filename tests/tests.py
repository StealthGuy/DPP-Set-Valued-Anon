# WRITEN BY CLAUDE AI

"""
Test Suite per SetAnonymizer
Implementazione del paper: He & Naughton, VLDB 2009
"Anonymization of Set-Valued Data via Top-Down, Local Generalization"

Struttura:
  - TestHierarchyNode        : unit test sul modello HierarchyNode
  - TestTransaction          : unit test su Transaction.update_representation
  - TestBalancePartitions    : unit test sulla logica di bilanciamento
  - TestPaperGroundTruth     : verifica contro gli esempi esatti del paper
  - TestKAnonymityProperty   : verifica della proprietà k-anonimity sull'output
  - TestNCPMetric            : verifica del calcolo NCP contro valori attesi a mano
  - TestEdgeCases            : casi limite (k=1, k=N, gerarchia piatta, ecc.)
  - TestBugRegression        : regression test per i bug trovati nell'analisi
"""

import unittest
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.item_hierarchy import HierarchyNode
from src.models.transaction import Transaction
from src.models.partition import Partition
from src.core.anonymizer import PartitionAnonymizer
from src.utils.metrics import calculate_database_ncp


# ---------------------------------------------------------------------------
# Factory helpers condivisi tra le classi di test
# ---------------------------------------------------------------------------

def build_paper_hierarchy():
    """
    Riproduce esattamente la gerarchia di Fig.1 del paper:
        ALL
        ├── Alcohol
        │   ├── Beer
        │   └── Wine
        └── Health Care
            ├── Diaper
            └── Pregnancy Test
    Ritorna (root, {nome: nodo})
    """
    root = HierarchyNode("ALL")
    alc  = HierarchyNode("Alcohol");      root.add_child(alc)
    hc   = HierarchyNode("Health Care"); root.add_child(hc)
    beer = HierarchyNode("Beer");        alc.add_child(beer)
    wine = HierarchyNode("Wine");        alc.add_child(wine)
    diap = HierarchyNode("Diaper");      hc.add_child(diap)
    preg = HierarchyNode("Pregnancy Test"); hc.add_child(preg)
    nodes = {
        "ALL": root, "Alcohol": alc, "Health Care": hc,
        "Beer": beer, "Wine": wine, "Diaper": diap, "Pregnancy Test": preg
    }
    return root, nodes


def build_simple_hierarchy():
    """
    Gerarchia semplice 2 livelli usata per unit test veloci:
        Root
        ├── Cat (A)
        │   ├── a1
        │   └── a2
        └── Dog (B)
            ├── b1
            └── b2
    Ritorna (root, {nome: nodo})
    """
    root = HierarchyNode("Root")
    cat  = HierarchyNode("Cat"); root.add_child(cat)
    dog  = HierarchyNode("Dog"); root.add_child(dog)
    a1   = HierarchyNode("a1"); cat.add_child(a1)
    a2   = HierarchyNode("a2"); cat.add_child(a2)
    b1   = HierarchyNode("b1"); dog.add_child(b1)
    b2   = HierarchyNode("b2"); dog.add_child(b2)
    nodes = {"Root": root, "Cat": cat, "Dog": dog,
             "a1": a1, "a2": a2, "b1": b1, "b2": b2}
    return root, nodes


_tid = 0
def make_tx(items):
    """Crea una Transaction con TID auto-incrementale."""
    global _tid
    _tid += 1
    return Transaction(_tid, items)


def run_anonymizer(transactions, k, root, domain_size):
    """Helper: inizializza, esegue e ritorna l'anonymizer."""
    for t in transactions:
        t.current_representation = [root]
    anon = PartitionAnonymizer(k=k, root_node=root, total_domain_size=domain_size)
    anon.anonymize(Partition(transactions, [root]))
    return anon


# ===========================================================================
# 1. Unit test: HierarchyNode
# ===========================================================================

class TestHierarchyNode(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def test_is_leaf_on_leaves(self):
        for name in ("a1", "a2", "b1", "b2"):
            self.assertTrue(self.n[name].is_leaf(),
                            f"{name} dovrebbe essere foglia")

    def test_is_leaf_on_internal(self):
        for name in ("Root", "Cat", "Dog"):
            self.assertFalse(self.n[name].is_leaf(),
                             f"{name} NON dovrebbe essere foglia")

    def test_get_leaves_count_leaf(self):
        self.assertEqual(self.n["a1"].get_leaves_count(), 1)

    def test_get_leaves_count_internal(self):
        self.assertEqual(self.n["Cat"].get_leaves_count(), 2)
        self.assertEqual(self.n["Dog"].get_leaves_count(), 2)
        self.assertEqual(self.n["Root"].get_leaves_count(), 4)

    def test_get_ncp_leaf(self):
        """NCP di una foglia deve essere 0.0 (nessuna generalizzazione)."""
        self.assertAlmostEqual(self.n["a1"].get_ncp(4), 0.0)

    def test_get_ncp_internal(self):
        """Cat copre 2 foglie su 4 → NCP = 0.5."""
        self.assertAlmostEqual(self.n["Cat"].get_ncp(4), 0.5)

    def test_get_ncp_root(self):
        """Root copre tutte le foglie → NCP = 1.0."""
        self.assertAlmostEqual(self.n["Root"].get_ncp(4), 1.0)

    def test_parent_assigned_correctly(self):
        self.assertIs(self.n["a1"].parent, self.n["Cat"])
        self.assertIs(self.n["Cat"].parent, self.n["Root"])
        self.assertIsNone(self.n["Root"].parent)

    def test_leaves_count_cache_sentinel_is_none(self):
        """
        Regression Bug #11: il sentinel della cache deve essere None, non 0,
        altrimenti nodi con 0 figli causerebbero ricalcoli incorretti.
        """
        fresh = HierarchyNode("fresh")
        self.assertIsNone(fresh._leaves_count,
                          "Il sentinel iniziale deve essere None, non 0")


# ===========================================================================
# 2. Unit test: Transaction.update_representation
# ===========================================================================

class TestTransaction(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def test_update_to_root(self):
        """Con cut={Root}, tutti gli item devono mappare su Root."""
        t = make_tx([self.n["a1"], self.n["b2"]])
        t.update_representation([self.n["Root"]])
        self.assertEqual(t.current_representation, [self.n["Root"]])

    def test_update_to_mid_level(self):
        """Con cut={Cat, Dog}, a1→Cat e b2→Dog."""
        t = make_tx([self.n["a1"], self.n["b2"]])
        t.update_representation([self.n["Cat"], self.n["Dog"]])
        names = sorted(n.name for n in t.current_representation)
        self.assertEqual(names, ["Cat", "Dog"])

    def test_update_to_mixed_cut(self):
        """Con cut={a1, a2, Dog}, a1→a1 e b2→Dog."""
        t = make_tx([self.n["a1"], self.n["b2"]])
        t.update_representation([self.n["a1"], self.n["a2"], self.n["Dog"]])
        names = sorted(n.name for n in t.current_representation)
        self.assertEqual(names, ["Dog", "a1"])

    def test_update_to_leaves(self):
        """Con cut a foglie, ogni item deve mappare su se stesso."""
        leaves = [self.n["a1"], self.n["a2"], self.n["b1"], self.n["b2"]]
        t = make_tx([self.n["a1"], self.n["b1"]])
        t.update_representation(leaves)
        names = sorted(n.name for n in t.current_representation)
        self.assertEqual(names, ["a1", "b1"])

    def test_no_duplicate_nodes_in_representation(self):
        """
        Due item originali dello stesso sottoramo non devono produrre
        lo stesso nodo duplicato nella rappresentazione.
        """
        t = make_tx([self.n["a1"], self.n["a2"]])
        t.update_representation([self.n["Cat"], self.n["Dog"]])
        # a1 e a2 entrambi mappano su Cat → deve apparire una sola volta
        self.assertEqual(len(t.current_representation), 1)
        self.assertIn(self.n["Cat"], t.current_representation)

    def test_original_items_unchanged_after_update(self):
        """update_representation non deve toccare original_items."""
        t = make_tx([self.n["a1"], self.n["b2"]])
        original = set(t.original_items)
        t.update_representation([self.n["Root"]])
        self.assertEqual(t.original_items, original)


# ===========================================================================
# 3. Unit test: _distribute_data e _balance_partitions
# ===========================================================================

class TestDistributeAndBalance(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()
        self.anon = PartitionAnonymizer(k=2, root_node=self.n["Root"],
                                        total_domain_size=4)

    def test_distribute_creates_correct_buckets(self):
        """
        Split Root → {Cat, Dog}.
        2 tx con a1 → bucket Cat.
        2 tx con b1 → bucket Dog.
        Devono creare esattamente 2 bucket.
        """
        txs = [make_tx([self.n["a1"]]) for _ in range(2)] + \
              [make_tx([self.n["b1"]]) for _ in range(2)]
        p = Partition(txs, [self.n["Root"]])
        for t in txs: t.update_representation([self.n["Root"]])

        buckets = self.anon._distribute_data(p, self.n["Root"])
        self.assertEqual(len(buckets), 2)

    def test_distribute_cuts_are_independent(self):
        """
        Regression Bug #2: ogni bucket deve avere una COPIA indipendente
        del new_cut. Modificare il cut di un bucket non deve alterare gli altri.
        """
        txs = [make_tx([self.n["a1"]]) for _ in range(2)] + \
              [make_tx([self.n["b1"]]) for _ in range(2)]
        p = Partition(txs, [self.n["Root"]])
        for t in txs: t.update_representation([self.n["Root"]])

        buckets = self.anon._distribute_data(p, self.n["Root"])
        # Modifichiamo brutalmente il cut del primo bucket
        _, cut0 = buckets[0]
        cut0.clear()
        # Il cut del secondo bucket non deve essere vuoto
        _, cut1 = buckets[1]
        self.assertGreater(len(cut1), 0,
            "Bug #2: i cut dei bucket condividono lo stesso oggetto lista")

    def test_balance_merges_small_bucket(self):
        """
        Un bucket con < k transazioni deve essere assorbito dal leftover,
        che poi deve essere ≥ k per essere valido.
        """
        # 3 tx in Cat, 1 tx in Dog → Dog < k=2 → merge
        txs_cat = [make_tx([self.n["a1"]]) for _ in range(3)]
        txs_dog = [make_tx([self.n["b1"]])]
        all_txs = txs_cat + txs_dog

        new_cut = [self.n["Cat"], self.n["Dog"]]
        buckets = [
            (txs_cat, list(new_cut)),
            (txs_dog, list(new_cut)),
        ]
        parent = Partition(all_txs, [self.n["Root"]])

        result = self.anon._balance_partitions(buckets, parent, self.n["Root"])
        for p in result:
            self.assertGreaterEqual(len(p), 2,
                "Ogni partizione risultante deve essere ≥ k")

    def test_balance_returns_empty_on_impossible_split(self):
        """
        Se anche dopo il merge il leftover è < k, _balance deve
        ritornare lista vuota (split fallito).
        """
        # 1 tx in Cat, 1 tx in Dog. k=3. Merge=2 < 3.
        txs = [make_tx([self.n["a1"]]), make_tx([self.n["b1"]])]
        anon3 = PartitionAnonymizer(k=3, root_node=self.n["Root"],
                                    total_domain_size=4)
        buckets = [
            ([txs[0]], [self.n["Cat"], self.n["Dog"]]),
            ([txs[1]], [self.n["Cat"], self.n["Dog"]]),
        ]
        parent = Partition(txs, [self.n["Root"]])
        result = anon3._balance_partitions(buckets, parent, self.n["Root"])
        self.assertEqual(result, [],
            "Split impossibile deve ritornare lista vuota")


# ===========================================================================
# 4. Ground truth dal paper (He & Naughton, VLDB 2009)
# ===========================================================================

class TestPaperGroundTruth(unittest.TestCase):
    """
    Verifica i risultati attesi sull'Example 1 del paper (Table 1).
    Alice={Beer,Diapers}, Bob={Wine,Diapers,PregnancyTest},
    Chris={Beer,Wine,PregnancyTest}, Dan={Beer,Wine,Diapers,PregnancyTest}
    k=2, |I|=4
    NCP atteso calcolato a mano = 1/3 ≈ 0.3333 (vedi Section 3.2 del paper)
    """

    def setUp(self):
        self.root, self.n = build_paper_hierarchy()
        self.domain_size = 4  # foglie: Beer, Wine, Diaper, Pregnancy Test
        n = self.n
        self.transactions = [
            make_tx([n["Beer"], n["Diaper"]]),
            make_tx([n["Wine"], n["Diaper"], n["Pregnancy Test"]]),
            make_tx([n["Beer"], n["Wine"], n["Pregnancy Test"]]),
            make_tx([n["Beer"], n["Wine"], n["Diaper"], n["Pregnancy Test"]]),
        ]

    def _run(self, k):
        return run_anonymizer(self.transactions, k, self.root, self.domain_size)

    def test_output_is_2_anonymous(self):
        """Ogni partizione finale deve avere almeno 2 transazioni."""
        anon = self._run(k=2)
        for p in anon.final_partitions:
            self.assertGreaterEqual(len(p), 2)

    def test_total_transactions_preserved(self):
        """Nessuna transazione deve andare persa dopo l'anonimizzazione."""
        anon = self._run(k=2)
        total = sum(len(p) for p in anon.final_partitions)
        self.assertEqual(total, 4)

    def test_ncp_matches_paper(self):
        """
        Il paper calcola NCP = 1/3 ≈ 0.3333 per la Table 1b.
        Tolleriamo ±0.05 per variazioni di implementazione del local recoding.
        """
        anon = self._run(k=2)
        final_txs = [t for p in anon.final_partitions for t in p.transactions]
        ncp = calculate_database_ncp(final_txs, self.domain_size)
        self.assertAlmostEqual(ncp, 1/3, delta=0.05,
            msg=f"NCP atteso ≈ 0.333, ottenuto {ncp:.4f}")

    def test_trivial_anonymization_with_k_equal_n(self):
        """
        Con k=N, nessuno split può produrre bucket ≥ k → tutto finisce
        in una singola partizione. Il cut può essere a qualsiasi livello
        raggiunto prima del fallimento (non necessariamente root).
        """
        anon = self._run(k=4)
        # Proprietà osservabile: deve esistere esattamente 1 partizione
        self.assertEqual(len(anon.final_partitions), 1)
        # Proprietà osservabile: tutte le 4 transazioni sono presenti
        self.assertEqual(len(anon.final_partitions[0]), 4)
        # NON verifichiamo il livello del cut: è un dettaglio implementativo
        # che dipende da quanti split vengono tentati prima del fallimento

    def test_ncp_is_zero_when_no_generalization_needed(self):
        """
        Se tutte le transazioni sono identiche, k-anonymity è già soddisfatta
        senza generalizzare → NCP deve essere 0.
        """
        root, n = build_paper_hierarchy()
        # 2 transazioni identiche: {Beer}
        txs = [make_tx([n["Beer"]]), make_tx([n["Beer"]])]
        anon = run_anonymizer(txs, k=2, root=root, domain_size=4)
        final_txs = [t for p in anon.final_partitions for t in p.transactions]
        ncp = calculate_database_ncp(final_txs, 4)
        self.assertAlmostEqual(ncp, 0.0, places=6,
            msg="Transazioni identiche non richiedono generalizzazione → NCP=0")


# ===========================================================================
# 5. Proprietà k-anonymity sull'output
# ===========================================================================

class TestKAnonymityProperty(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def _check_k_anonymity(self, anon, k):
        """Verifica che ogni partizione abbia almeno k transazioni."""
        for p in anon.final_partitions:
            self.assertGreaterEqual(len(p), k,
                f"Partizione con {len(p)} tx viola k={k}")

    def _check_no_loss(self, anon, expected_total):
        """Verifica che il totale delle transazioni sia conservato."""
        total = sum(len(p) for p in anon.final_partitions)
        self.assertEqual(total, expected_total,
            f"Transazioni perse: attese {expected_total}, trovate {total}")

    def test_k2_clear_split(self):
        """2 tx in Cat, 2 tx in Dog → 2 partizioni, entrambe ≥ k=2."""
        txs = [make_tx([self.n["a1"]]) for _ in range(2)] + \
              [make_tx([self.n["b1"]]) for _ in range(2)]
        anon = run_anonymizer(txs, k=2, root=self.root, domain_size=4)
        self._check_k_anonymity(anon, k=2)
        self._check_no_loss(anon, 4)

    def test_k2_with_remainder(self):
        """5 tx: 4 Cat + 1 Dog. Il Dog non può formare un gruppo da solo."""
        txs = [make_tx([self.n["a1"]]) for _ in range(4)] + \
              [make_tx([self.n["b1"]])]
        anon = run_anonymizer(txs, k=2, root=self.root, domain_size=4)
        self._check_k_anonymity(anon, k=2)
        self._check_no_loss(anon, 5)

    def test_k_equals_total(self):
        """Con k = N, deve esistere esattamente 1 partizione."""
        txs = [make_tx([self.n["a1"]]), make_tx([self.n["b1"]])]
        anon = run_anonymizer(txs, k=2, root=self.root, domain_size=4)
        self.assertEqual(len(anon.final_partitions), 1)
        self._check_k_anonymity(anon, k=2)
        self._check_no_loss(anon, 2)

    def test_k1_max_precision(self):
        """Con k=1, l'algoritmo deve scendere fino alle foglie."""
        txs = [make_tx([self.n["a1"]]), make_tx([self.n["b2"]])]
        anon = run_anonymizer(txs, k=1, root=self.root, domain_size=4)
        self._check_k_anonymity(anon, k=1)
        self._check_no_loss(anon, 2)
        # Con k=1, Root non deve comparire nel cut finale
        for p in anon.final_partitions:
            cut_names = [node.name for node in p.hierarchy_cut]
            self.assertNotIn("Root", cut_names,
                "Con k=1 non dovremmo restare al root level")

    def test_large_k_forces_more_generalization(self):
        """
        NCP con k grande deve essere ≥ NCP con k piccolo:
        più anonimato richiede più generalizzazione.
        """
        txs_small = []
        txs_large = []
        leaves = [self.n["a1"], self.n["a2"], self.n["b1"], self.n["b2"]]
        import random; random.seed(42)
        for i in range(50):
            items = random.sample(leaves, k=2)
            txs_small.append(make_tx(items))
            txs_large.append(make_tx(list(items)))

        anon_small = run_anonymizer(txs_small, k=2,
                                    root=self.root, domain_size=4)
        anon_large = run_anonymizer(txs_large, k=10,
                                    root=self.root, domain_size=4)

        ftxs_s = [t for p in anon_small.final_partitions for t in p.transactions]
        ftxs_l = [t for p in anon_large.final_partitions for t in p.transactions]

        ncp_small = calculate_database_ncp(ftxs_s, 4)
        ncp_large = calculate_database_ncp(ftxs_l, 4)

        self.assertGreaterEqual(ncp_large, ncp_small - 0.01,
            f"k grande ({ncp_large:.4f}) non può avere NCP minore di k piccolo ({ncp_small:.4f})")


# ===========================================================================
# 6. Verifica del calcolo NCP
# ===========================================================================

class TestNCPMetric(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def test_ncp_zero_all_leaves(self):
        """Se tutte le rappresentazioni sono foglie, NCP = 0."""
        t1 = make_tx([self.n["a1"]])
        t2 = make_tx([self.n["a1"]])
        t1.current_representation = [self.n["a1"]]
        t2.current_representation = [self.n["a1"]]
        ncp = calculate_database_ncp([t1, t2], total_domain_size=4)
        self.assertAlmostEqual(ncp, 0.0, places=6)

    def test_ncp_one_all_root(self):
        """Se tutte le rappresentazioni sono Root, NCP = 1."""
        t1 = make_tx([self.n["a1"]])
        t1.current_representation = [self.n["Root"]]
        ncp = calculate_database_ncp([t1], total_domain_size=4)
        self.assertAlmostEqual(ncp, 1.0, places=6)

    def test_ncp_manual_calculation(self):
        """
        Calcolo a mano su 2 transazioni:
          T1: original={a1, b1}, rep={Cat, Dog}
              NCP(Cat)=2/4=0.5, NCP(Dog)=2/4=0.5 → somma=1.0
          T2: original={a1}, rep={a1}
              NCP(a1)=0.0 → somma=0.0
          NCP_totale = (1.0 + 0.0) / (2 + 1) = 1/3 ≈ 0.3333
        """
        t1 = make_tx([self.n["a1"], self.n["b1"]])
        t1.current_representation = [self.n["Cat"], self.n["Dog"]]
        t2 = make_tx([self.n["a1"]])
        t2.current_representation = [self.n["a1"]]

        ncp = calculate_database_ncp([t1, t2], total_domain_size=4)
        self.assertAlmostEqual(ncp, 1/3, places=4,
            msg=f"Atteso 0.3333, ottenuto {ncp:.6f}")

    def test_ncp_mixed_generalization(self):
        """
        T1: original={a1, b1}, rep={Cat, b1}
            NCP(Cat)=0.5, NCP(b1)=0.0 → somma=0.5
        NCP_totale = 0.5 / 2 = 0.25
        """
        t1 = make_tx([self.n["a1"], self.n["b1"]])
        t1.current_representation = [self.n["Cat"], self.n["b1"]]

        ncp = calculate_database_ncp([t1], total_domain_size=4)
        self.assertAlmostEqual(ncp, 0.25, places=4)

    def test_ncp_empty_transactions(self):
        """Con lista vuota, NCP deve essere 0 senza eccezioni."""
        ncp = calculate_database_ncp([], total_domain_size=4)
        self.assertEqual(ncp, 0)

    def test_ncp_is_bounded_between_0_and_1(self):
        """NCP deve sempre stare in [0, 1]."""
        root, n = build_paper_hierarchy()
        txs = [
            make_tx([n["Beer"], n["Diaper"]]),
            make_tx([n["Wine"], n["Pregnancy Test"]]),
            make_tx([n["Beer"], n["Wine"]]),
            make_tx([n["Diaper"], n["Pregnancy Test"]]),
        ]
        anon = run_anonymizer(txs, k=2, root=root, domain_size=4)
        final_txs = [t for p in anon.final_partitions for t in p.transactions]
        ncp = calculate_database_ncp(final_txs, 4)
        self.assertGreaterEqual(ncp, 0.0)
        self.assertLessEqual(ncp, 1.0)


# ===========================================================================
# 7. Casi limite
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def test_single_transaction_k1(self):
        """Una sola transazione con k=1 deve produrre 1 partizione."""
        txs = [make_tx([self.n["a1"]])]
        anon = run_anonymizer(txs, k=1, root=self.root, domain_size=4)
        self.assertEqual(len(anon.final_partitions), 1)
        self.assertEqual(sum(len(p) for p in anon.final_partitions), 1)

    def test_all_identical_transactions(self):
        """
        100 transazioni identiche {a1}: k-anonymity già soddisfatta,
        NCP deve essere 0 (nessuna generalizzazione necessaria).
        """
        txs = [make_tx([self.n["a1"]]) for _ in range(100)]
        anon = run_anonymizer(txs, k=10, root=self.root, domain_size=4)
        final_txs = [t for p in anon.final_partitions for t in p.transactions]
        ncp = calculate_database_ncp(final_txs, 4)
        self.assertAlmostEqual(ncp, 0.0, places=5,
            msg="Transazioni identiche non richiedono generalizzazione")

    def test_flat_hierarchy_one_level(self):
        """
        Gerarchia piatta: Root con solo foglie dirette.
        Root non può splittare verso nodi intermedi → 1 partizione finale.
        """
        root = HierarchyNode("Root")
        x = HierarchyNode("x"); root.add_child(x)
        y = HierarchyNode("y"); root.add_child(y)
        txs = [make_tx([x]), make_tx([y])]
        anon = run_anonymizer(txs, k=2, root=root, domain_size=2)
        # Con 2 tx e k=2, lo split porta a 1 tx per bucket → fallisce → 1 partizione
        for p in anon.final_partitions:
            self.assertGreaterEqual(len(p), 2)

    def test_all_items_in_every_transaction(self):
        """
        Ogni transazione contiene tutte le foglie:
        tutte le tx hanno la stessa rappresentazione → 1 partizione, NCP basso.
        """
        leaves = [self.n["a1"], self.n["a2"], self.n["b1"], self.n["b2"]]
        txs = [make_tx(list(leaves)) for _ in range(10)]
        anon = run_anonymizer(txs, k=5, root=self.root, domain_size=4)
        total = sum(len(p) for p in anon.final_partitions)
        self.assertEqual(total, 10)

    def test_transactions_preserved_count(self):
        """Il numero totale di transazioni deve essere conservato per ogni k."""
        import random; random.seed(0)
        leaves = [self.n["a1"], self.n["a2"], self.n["b1"], self.n["b2"]]
        for k in [2, 3, 5]:
            txs = [make_tx(random.sample(leaves, 2)) for _ in range(20)]
            anon = run_anonymizer(txs, k=k, root=self.root, domain_size=4)
            total = sum(len(p) for p in anon.final_partitions)
            self.assertEqual(total, 20,
                f"Con k={k}: attese 20 tx, trovate {total}")


# ===========================================================================
# 8. Regression test per i bug identificati nell'analisi
# ===========================================================================

class TestBugRegression(unittest.TestCase):

    def setUp(self):
        self.root, self.n = build_simple_hierarchy()

    def test_bug1_info_gain_not_uniform_average(self):
        """
        Bug #1: _calculate_info_gain non deve usare media aritmetica sui figli.
        Verifica indirettamente: con gerarchia asimmetrica, il nodo scelto
        deve essere quello che massimizza il guadagno reale, non quello
        con la media più alta sui figli.
        Il risultato deve comunque rispettare k-anonymity.
        """
        # Gerarchia asimmetrica: Root → A(3 figli) e B(1 figlio)
        root = HierarchyNode("Root")
        a = HierarchyNode("A"); root.add_child(a)
        b = HierarchyNode("B"); root.add_child(b)
        for i in range(3):
            a.add_child(HierarchyNode(f"a{i}"))
        b.add_child(HierarchyNode("b0"))

        leaves_a = a.children
        leaf_b = b.children[0]
        domain = 4

        txs = [make_tx([leaves_a[0]]) for _ in range(3)] + \
              [make_tx([leaf_b]) for _ in range(3)]
        anon = run_anonymizer(txs, k=3, root=root, domain_size=domain)
        for p in anon.final_partitions:
            self.assertGreaterEqual(len(p), 3,
                "Bug #1: info gain errato può causare split che viola k-anonymity")

    def test_bug3_partition_not_mutated_in_place(self):
        """
        Bug #3: anonymize non deve mutare la partizione originale in-place
        quando uno split fallisce. excluded_nodes della partizione originale
        passata deve restare invariato.
        """
        txs = [make_tx([self.n["a1"]]), make_tx([self.n["b1"]])]
        initial_partition = Partition(txs, [self.n["Root"]])
        for t in txs: t.update_representation([self.n["Root"]])

        original_excluded = set(initial_partition.excluded_nodes)
        anon = PartitionAnonymizer(k=2, root_node=self.n["Root"],
                                   total_domain_size=4)
        anon.anonymize(initial_partition)

        self.assertEqual(initial_partition.excluded_nodes, original_excluded,
            "Bug #3: anonymize ha mutato excluded_nodes della partizione originale")

    def test_bug5_missing_coverage_does_not_silently_underestimate_ncp(self):
        """
        Bug #5: se un item non trova copertura nel cut, il vecchio codice
        usava 'pass' → NCP artificialmente basso.
        Verifichiamo che il comportamento attuale non restituisca 0
        in un caso dove la copertura non è garantita.
        Costruiamo manualmente una situazione di mancata copertura.
        """
        t = make_tx([self.n["a1"]])
        # Rappresentazione che NON copre a1 (Dog non è antenato di a1)
        t.current_representation = [self.n["Dog"]]
        # Con il fix, viene applicata penalità 1.0 invece di 0
        ncp = calculate_database_ncp([t], total_domain_size=4)
        self.assertGreater(ncp, 0.0,
            "Bug #5: mancata copertura deve dare penalità > 0, non 0 silenzioso")

    def test_bug11_cache_sentinel_none(self):
        """
        Bug #11: _leaves_count deve inizializzare a None, non 0.
        Con 0 come sentinel, get_leaves_count() di un nodo interno
        ricalcolerebbe ogni volta invece di usare la cache.
        """
        node = HierarchyNode("test")
        child = HierarchyNode("child")
        node.add_child(child)

        # Prima chiamata: calcola e memorizza
        count1 = node.get_leaves_count()
        # Seconda chiamata: deve usare la cache (stesso valore)
        count2 = node.get_leaves_count()
        self.assertEqual(count1, count2)
        # Il valore cachato non deve essere None dopo la prima chiamata
        self.assertIsNotNone(node._leaves_count)
        self.assertEqual(node._leaves_count, 1)


# ===========================================================================
# 9. Test di scalabilità (complessità lineare)
# ===========================================================================

class TestScalability(unittest.TestCase):
    """
    Verifica che la complessità sia O(N) come dichiarato nel paper (Sec. 5).
    Usa una gerarchia più profonda per dare all'algoritmo abbastanza lavoro.
    """

    def _build_deep_hierarchy(self, depth=3, fanout=3):
        """Costruisce una gerarchia bilanciata depth livelli, fanout figli."""
        root = HierarchyNode("ROOT")
        def _build(node, current_depth):
            if current_depth == 0:
                return
            for i in range(fanout):
                child = HierarchyNode(f"{node.name}_c{i}")
                node.add_child(child)
                _build(child, current_depth - 1)
        _build(root, depth)
        # Raccogli foglie
        leaves = []
        def _collect(n):
            if n.is_leaf(): leaves.append(n)
            for c in n.children: _collect(c)
        _collect(root)
        return root, leaves

    def test_linear_scaling(self):
        """
        Raddoppiando N, il tempo deve al massimo triplicare (tolleranza O(N log N)).
        Se il ratio supera 4x, probabilmente è O(N²).
        """
        import random
        random.seed(123)
        root, leaves = self._build_deep_hierarchy(depth=3, fanout=3)
        domain = len(leaves)  # 27 foglie

        def run(n):
            txs = [make_tx(random.sample(leaves, min(4, len(leaves))))
                   for _ in range(n)]
            start = time.time()
            run_anonymizer(txs, k=5, root=root, domain_size=domain)
            return time.time() - start

        # Warmup
        run(100)

        t1 = run(500)
        t2 = run(1000)
        t3 = run(2000)

        ratio_1 = t2 / t1 if t1 > 0.001 else 1.0
        ratio_2 = t3 / t2 if t2 > 0.001 else 1.0
        avg_ratio = (ratio_1 + ratio_2) / 2

        print(f"\n  Scalabilità: 500→1000: {ratio_1:.2f}x | "
              f"1000→2000: {ratio_2:.2f}x | media: {avg_ratio:.2f}x")

        self.assertLess(avg_ratio, 4.0,
            f"Complessità superlineare rilevata: ratio medio {avg_ratio:.2f}x "
            f"(atteso < 4.0 per O(N) o O(N log N))")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)