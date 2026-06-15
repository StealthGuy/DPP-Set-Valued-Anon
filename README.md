# Set-Valued Data Anonymizer (He & Naughton, VLDB 2009)

This project implements the top-down local generalization algorithm for set-valued data anonymization proposed by He & Naughton (VLDB 2009). It solves the privacy protection problem on sparse, high-dimensional transactional databases by partitioning transactions and applying localized generalization using a taxonomy hierarchy tree to achieve k-anonymity. The implementation was developed as part of the Data Protection and Privacy (DPP) course at the University of Genova (UniGe).

Refer to the presentation slides for a overview and experimental analysis: [Presentation Slides](DPP_compressed.pdf).

## Project Structure

```
├── data/
│   └── raw/                # JSON hierarchy/taxonomy files (e.g., good_hierarchy.json)
├── src/
│   ├── core/               # PartitionAnonymizer algorithm implementation
│   ├── models/             # Data structures (HierarchyNode, Transaction, Partition)
│   ├── utils/              # NCP metrics, synthetic generator, JSON hierarchy loaders
│   ├── main.py             # CLI entrypoint to run anonymization
│   └── run_analysis.py     # Batch analysis suite generating experimental plots
├── tests/
│   ├── tests.py            # Main test suite (unit, integration, regression, scaling)
│   └── verify_fixes.py     # Bug regression verification scripts
├── requirements.txt        # Minimal library dependencies
└── DPP_compressed.pdf      # Course presentation slides
```

## Installation

Ensure you have Python 3 installed. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

### 1. Run Unit and Integration Tests
Verify the correctness of the models, metrics, and anonymizer split logic:
```bash
python3 tests/tests.py
```

### 2. Run Main CLI
Run the anonymizer on a default paper scenario (Alice, Bob, Chris, Dan transactions):
```bash
python3 src/main.py
```

Or run it on synthetic data generated from a custom hierarchy, configuring the number of transactions and the k-anonymity constraint:
```bash
python3 src/main.py --json data/raw/good_hierarchy.json -k 10 --num 1000 --save_csv data/synthetic/generated.csv --output data/synthetic/anonymized.csv
```

CLI parameters:
* `--json`: Path to JSON item taxonomy hierarchy.
* `-k`: Privacy constraint parameter (default: 2).
* `--num`: Count of generated synthetic transactions (default: 20).
* `--avg_size`: Average items per transaction cart (default: 4).
* `--bias`: Category bias parameter (0.0 to 1.0) for the synthetic generator (default: 0.7).
* `--corr`: Sibling item local correlation strength (0.0 to 1.0) (default: 0.5).
* `--save_csv`: Output file path to save the generated synthetic dataset.
* `--output` / `-o`: Output file path to save the generalized k-anonymized transaction database.

### 3. Generate Experimental Charts
Run the automated analysis script to reproduce the paper's experiments (NCP vs k, scalability runtime vs |D|, hierarchy fan-out sensitivity, and Pareto trade-off curves):
```bash
python3 src/run_analysis.py
```
This saves 10 plots directly into the `src/analysis_results/` folder.

## References

* He, Y., & Naughton, J. F. (2009). Anonymization of set-valued data via top-down, local generalization. *Proceedings of the VLDB Endowment*, 2(1), 934-945.
