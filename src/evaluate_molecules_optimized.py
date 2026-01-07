"""
Evaluation script for generated molecules - Optimized Version
Computes: Validity, Uniqueness, Novelty, Scaff, FCD, Filter, SNN, IntDiv, IntDiv2, Frag
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter
import warnings
import random
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Try to import torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import FilterCatalog for PAINS
try:
    from rdkit.Chem import FilterCatalog
    FILTERCATALOG_AVAILABLE = True
except ImportError:
    FILTERCATALOG_AVAILABLE = False

try:
    from moses.metrics import get_all_metrics
    MOSES_AVAILABLE = True
except ImportError:
    MOSES_AVAILABLE = False
    print("Warning: MOSES library not available. Some metrics will use alternative implementations.")

try:
    from fcd import get_fcd
    FCD_AVAILABLE = True
except ImportError:
    FCD_AVAILABLE = False
    print("Warning: FCD library not available. FCD metric will be skipped. Install with: pip install fcd-torch")


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def load_smiles_from_file(filepath: str, canonicalize: bool = True) -> List[str]:
    """Load SMILES strings from a text file (one per line)."""
    smiles_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.lower() != 'none':
                if canonicalize:
                    canon_smiles = canonicalize_smiles(line)
                    if canon_smiles:
                        smiles_list.append(canon_smiles)
                else:
                    smiles_list.append(line)
    return smiles_list


def validate_single_smiles(smiles: str) -> Optional[str]:
    """Validate a single SMILES string. Returns canonical SMILES if valid, None otherwise."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def compute_validity(smiles_list: List[str], n_jobs: int = -1) -> Tuple[float, List[str]]:
    """Compute validity: fraction of valid SMILES. Supports parallel processing."""
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # Use parallel processing for large datasets
    if len(smiles_list) > 1000 and n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            valid_smiles = [s for s in pool.map(validate_single_smiles, smiles_list) if s is not None]
    else:
        valid_smiles = [s for s in [validate_single_smiles(s) for s in smiles_list] if s is not None]
    
    validity = len(valid_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0.0
    return validity, valid_smiles


def compute_uniqueness(valid_smiles: List[str]) -> Tuple[float, List[str]]:
    """Compute uniqueness: fraction of unique valid SMILES."""
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0.0
    return uniqueness, unique_smiles


def compute_novelty(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> float:
    """Compute novelty: fraction of unique SMILES not in reference set."""
    if reference_smiles is None:
        return -1.0
    
    reference_set = set(reference_smiles)
    novel_count = sum(1 for s in unique_smiles if s not in reference_set)
    novelty = novel_count / len(unique_smiles) if len(unique_smiles) > 0 else 0.0
    return novelty


def extract_scaffold(smiles: str) -> Optional[str]:
    """Extract Murcko scaffold from a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
    except:
        pass
    return None


def compute_scaffold_similarity(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None, n_jobs: int = -1) -> float:
    """Compute scaffold similarity: fraction of generated scaffolds present in reference set."""
    if reference_smiles is None:
        return -1.0
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    # Parallel processing for scaffold extraction
    if len(unique_smiles) > 500 and n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            gen_scaffolds = set(s for s in pool.map(extract_scaffold, unique_smiles) if s is not None)
            ref_scaffolds = set(s for s in pool.map(extract_scaffold, reference_smiles) if s is not None)
    else:
        gen_scaffolds = set(s for s in map(extract_scaffold, unique_smiles) if s is not None)
        ref_scaffolds = set(s for s in map(extract_scaffold, reference_smiles) if s is not None)
    
    # Compute intersection
    common_scaffolds = gen_scaffolds.intersection(ref_scaffolds)
    scaff = len(common_scaffolds) / len(gen_scaffolds) if len(gen_scaffolds) > 0 else 0.0
    return scaff


def compute_fcd(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> float:
    """Compute FCD (Fréchet ChemNet Distance)."""
    if reference_smiles is None:
        return -1.0
    
    if not FCD_AVAILABLE:
        return -1.0
    
    try:
        fcd_value = get_fcd(unique_smiles, reference_smiles)
        return fcd_value
    except Exception as e:
        print(f"Error computing FCD: {e}")
        return -1.0


# Initialize PAINS filter catalog (lazy loading)
_pains_catalog = None

def get_pains_catalog():
    """Get or create PAINS filter catalog."""
    global _pains_catalog
    if _pains_catalog is None and FILTERCATALOG_AVAILABLE:
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            _pains_catalog = FilterCatalog.FilterCatalog(params)
        except:
            _pains_catalog = False  # Mark as unavailable
    return _pains_catalog


def has_pains(mol) -> bool:
    """Check for PAINS (Pan Assay Interference Compounds) using FilterCatalog."""
    catalog = get_pains_catalog()
    if catalog:
        try:
            entry = catalog.GetFirstMatch(mol)
            return entry is not None
        except:
            pass
    
    # Fallback to simple pattern matching
    pains_patterns = [
        '[N+](=O)[O-]',  # Nitro group
        '[S,P](=O)(=O)',  # Sulfonyl/Phosphonyl
    ]
    for pattern in pains_patterns:
        try:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        except:
            continue
    return False


def check_filter(smiles: str) -> bool:
    """Check if a molecule passes filters."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
            if not has_pains(mol):
                return True
    except:
        pass
    return False


def compute_filter(unique_smiles: List[str], n_jobs: int = -1) -> float:
    """Compute Filter: fraction of molecules passing MOSES filters."""
    if MOSES_AVAILABLE:
        try:
            device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            metrics = get_all_metrics(
                gen=unique_smiles,
                train=None,
                test=None,
                device=device,
                n_jobs=n_jobs
            )
            return metrics.get('Filters', -1.0)
        except Exception as e:
            print(f"Error computing Filter with MOSES: {e}")
            # Fall through to alternative implementation
    
    # Alternative implementation using basic filters
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    if len(unique_smiles) > 1000 and n_jobs > 1:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(check_filter, unique_smiles)
            passed = sum(results)
    else:
        passed = sum(check_filter(s) for s in unique_smiles)
    
    return passed / len(unique_smiles) if len(unique_smiles) > 0 else 0.0


def compute_snn_batch(gen_smiles_batch: List[str], ref_fps: List) -> List[float]:
    """Compute SNN for a batch of generated SMILES."""
    similarities = []
    for gen_smiles in gen_smiles_batch:
        try:
            gen_mol = Chem.MolFromSmiles(gen_smiles)
            if gen_mol is None:
                continue
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
            
            max_sim = 0.0
            for ref_fp in ref_fps:
                try:
                    sim = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
                    max_sim = max(max_sim, sim)
                except:
                    continue
            similarities.append(max_sim)
        except:
            continue
    return similarities


def compute_snn(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None, 
                batch_size: int = 1000, n_jobs: int = -1) -> float:
    """Compute SNN (Similarity to Nearest Neighbor) with batch processing."""
    if reference_smiles is None:
        return -1.0
    
    # Try to use MOSES implementation first
    if MOSES_AVAILABLE:
        try:
            device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            metrics = get_all_metrics(
                gen=unique_smiles,
                train=reference_smiles,
                test=None,
                device=device,
                n_jobs=n_jobs
            )
            return metrics.get('SNN', -1.0)
        except Exception as e:
            print(f"Error computing SNN with MOSES: {e}")
            # Fall through to alternative implementation
    
    # Alternative implementation using Tanimoto similarity with batch processing
    # Pre-compute reference fingerprints for efficiency
    print(f"    Pre-computing reference fingerprints ({len(reference_smiles)} molecules)...")
    ref_fps = []
    for ref_smiles in tqdm(reference_smiles, desc="    Computing ref fingerprints", leave=False):
        try:
            ref_mol = Chem.MolFromSmiles(ref_smiles)
            if ref_mol is not None:
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                ref_fps.append(ref_fp)
        except:
            continue
    
    if len(ref_fps) == 0:
        return -1.0
    
    # Batch processing to avoid memory issues
    all_similarities = []
    for i in tqdm(range(0, len(unique_smiles), batch_size), desc="    Computing SNN"):
        batch = unique_smiles[i:i+batch_size]
        batch_similarities = compute_snn_batch(batch, ref_fps)
        all_similarities.extend(batch_similarities)
    
    return np.mean(all_similarities) if len(all_similarities) > 0 else -1.0


def compute_internal_diversity(smiles_list: List[str], p: int = 1) -> float:
    """Compute Internal Diversity (IntDiv).
    
    Args:
        smiles_list: List of SMILES strings
        p: Power for diversity metric (1 for IntDiv, 2 for IntDiv2)
    """
    if len(smiles_list) < 2:
        return 0.0
    
    similarities = []
    fps = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
        except:
            continue
    
    if len(fps) < 2:
        return 0.0
    
    # Compute pairwise similarities (sample if too many to avoid memory issues)
    max_pairs = 10000  # Limit number of pairs to compute
    n_fps = len(fps)
    total_pairs = n_fps * (n_fps - 1) // 2
    
    if total_pairs > max_pairs:
        # Sample pairs randomly
        pairs_to_compute = random.sample(
            [(i, j) for i in range(n_fps) for j in range(i + 1, n_fps)],
            min(max_pairs, total_pairs)
        )
    else:
        pairs_to_compute = [(i, j) for i in range(n_fps) for j in range(i + 1, n_fps)]
    
    for i, j in pairs_to_compute:
        try:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
        except:
            continue
    
    if len(similarities) == 0:
        return 0.0
    
    # IntDiv = 1 - mean(similarities)
    # IntDiv2 = 1 - sqrt(mean(similarities^2))
    if p == 1:
        return 1.0 - np.mean(similarities)
    elif p == 2:
        mean_sim_sq = np.mean([s**2 for s in similarities])
        return 1.0 - np.sqrt(mean_sim_sq)
    else:
        raise ValueError(f"Invalid p value: {p}. Must be 1 or 2.")


def get_fragments_brics(smiles: str) -> List[str]:
    """Extract molecular fragments using BRICS decomposition."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Use BRICS to break into fragments
            # Note: BRICSDecompose returns a generator, convert to list
            frags = Chem.rdmolops.BRICSDecompose(mol)
            # Convert fragments to SMILES
            fragment_smiles = []
            for frag in frags:
                try:
                    frag_smiles = Chem.MolToSmiles(frag, canonical=True)
                    fragment_smiles.append(frag_smiles)
                except:
                    continue
            return fragment_smiles
    except:
        pass
    return []


def compute_fragment_similarity(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None, n_jobs: int = -1) -> float:
    """Compute Fragment Similarity: similarity of fragment distributions using BRICS."""
    if reference_smiles is None:
        return -1.0
    
    # Try to use MOSES implementation first
    if MOSES_AVAILABLE:
        try:
            device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            metrics = get_all_metrics(
                gen=unique_smiles,
                train=reference_smiles,
                test=None,
                device=device,
                n_jobs=n_jobs
            )
            frag_value = metrics.get('Frag', -1.0)
            if frag_value != -1.0:
                return frag_value
        except Exception as e:
            print(f"Error computing Frag with MOSES: {e}")
            # Fall through to alternative implementation
    
    # Alternative implementation using BRICS fragments
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    def get_fragments(smiles_list):
        """Extract molecular fragments (BRICS fragments) with parallel processing."""
        fragments = []
        if len(smiles_list) > 500 and n_jobs > 1:
            with mp.Pool(n_jobs) as pool:
                fragment_lists = pool.map(get_fragments_brics, smiles_list)
                fragments = [f for frag_list in fragment_lists for f in frag_list]
        else:
            for smiles in smiles_list:
                fragments.extend(get_fragments_brics(smiles))
        return Counter(fragments)
    
    gen_fragments = get_fragments(unique_smiles)
    ref_fragments = get_fragments(reference_smiles)
    
    if len(gen_fragments) == 0 or len(ref_fragments) == 0:
        return -1.0
    
    # Compute Jaccard similarity of fragment sets
    gen_set = set(gen_fragments.keys())
    ref_set = set(ref_fragments.keys())
    
    intersection = gen_set.intersection(ref_set)
    union = gen_set.union(ref_set)
    
    frag_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
    return frag_sim


def evaluate_molecules(generated_file: str, reference_file: Optional[str] = None, 
                      n_jobs: int = -1, batch_size: int = 1000, use_moses_direct: bool = True) -> dict:
    """
    Evaluate generated molecules against reference set.
    
    Args:
        generated_file: Path to file with generated SMILES (one per line)
        reference_file: Optional path to file with reference SMILES (for Novelty, Scaff, FCD, SNN, Frag)
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        batch_size: Batch size for SNN computation
        use_moses_direct: Whether to use MOSES get_all_metrics directly (faster if available)
    
    Returns:
        Dictionary with all computed metrics
    """
    print(f"Loading generated molecules from: {generated_file}")
    generated_smiles = load_smiles_from_file(generated_file)
    print(f"Loaded {len(generated_smiles)} generated molecules")
    
    reference_smiles = None
    if reference_file and os.path.exists(reference_file):
        print(f"Loading reference molecules from: {reference_file}")
        reference_smiles = load_smiles_from_file(reference_file)
        print(f"Loaded {len(reference_smiles)} reference molecules")
    
    # Use MOSES directly if available and requested
    if use_moses_direct and MOSES_AVAILABLE and reference_smiles is not None:
        print("\nUsing MOSES metrics directly (faster)...")
        try:
            device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
            if n_jobs == -1:
                n_jobs = mp.cpu_count()
            
            # Compute basic metrics first
            print("  Computing Validity...")
            validity, valid_smiles = compute_validity(generated_smiles, n_jobs=n_jobs)
            
            print("  Computing Uniqueness...")
            uniqueness, unique_smiles = compute_uniqueness(valid_smiles)
            
            # Get all metrics from MOSES
            print("  Computing MOSES metrics (this may take a while)...")
            moses_metrics = get_all_metrics(
                gen=unique_smiles,
                train=reference_smiles,
                test=None,
                device=device,
                n_jobs=n_jobs
            )
            
            # Extract metrics
            novelty = moses_metrics.get('Novelty', -1.0)
            scaff = moses_metrics.get('Scaffold', -1.0)
            fcd = moses_metrics.get('FCD', -1.0)
            filter_metric = moses_metrics.get('Filters', -1.0)
            snn = moses_metrics.get('SNN', -1.0)
            intdiv = moses_metrics.get('IntDiv', -1.0)
            intdiv2 = moses_metrics.get('IntDiv2', -1.0)
            frag = moses_metrics.get('Frag', -1.0)
            
            # If some metrics are missing, compute them separately
            if intdiv == -1.0:
                print("  Computing IntDiv...")
                intdiv = compute_internal_diversity(unique_smiles, p=1)
            if intdiv2 == -1.0:
                print("  Computing IntDiv2...")
                intdiv2 = compute_internal_diversity(unique_smiles, p=2)
            if frag == -1.0:
                print("  Computing Frag...")
                frag = compute_fragment_similarity(unique_smiles, reference_smiles, n_jobs=n_jobs)
            if scaff == -1.0:
                print("  Computing Scaff...")
                scaff = compute_scaffold_similarity(unique_smiles, reference_smiles, n_jobs=n_jobs)
            if fcd == -1.0:
                print("  Computing FCD...")
                fcd = compute_fcd(unique_smiles, reference_smiles)
            
        except Exception as e:
            print(f"Error using MOSES metrics directly: {e}")
            print("Falling back to individual metric computation...")
            use_moses_direct = False
    
    if not use_moses_direct or not MOSES_AVAILABLE:
        # Compute metrics individually
        print("\nComputing metrics...")
        
        # Validity
        print("  Computing Validity...")
        validity, valid_smiles = compute_validity(generated_smiles, n_jobs=n_jobs)
        
        # Uniqueness
        print("  Computing Uniqueness...")
        uniqueness, unique_smiles = compute_uniqueness(valid_smiles)
        
        # Novelty
        print("  Computing Novelty...")
        novelty = compute_novelty(unique_smiles, reference_smiles)
        
        # Scaff
        print("  Computing Scaff...")
        scaff = compute_scaffold_similarity(unique_smiles, reference_smiles, n_jobs=n_jobs)
        
        # FCD
        print("  Computing FCD...")
        fcd = compute_fcd(unique_smiles, reference_smiles)
        
        # Filter
        print("  Computing Filter...")
        filter_metric = compute_filter(unique_smiles, n_jobs=n_jobs)
        
        # SNN
        print("  Computing SNN...")
        snn = compute_snn(unique_smiles, reference_smiles, batch_size=batch_size, n_jobs=n_jobs)
        
        # IntDiv
        print("  Computing IntDiv...")
        intdiv = compute_internal_diversity(unique_smiles, p=1)
        
        # IntDiv2
        print("  Computing IntDiv2...")
        intdiv2 = compute_internal_diversity(unique_smiles, p=2)
        
        # Frag
        print("  Computing Frag...")
        frag = compute_fragment_similarity(unique_smiles, reference_smiles, n_jobs=n_jobs)
    
    results = {
        'Validity': validity,
        'Uniqueness': uniqueness,
        'Novelty': novelty,
        'Scaff': scaff,
        'FCD': fcd,
        'Filter': filter_metric,
        'SNN': snn,
        'IntDiv': intdiv,
        'IntDiv2': intdiv2,
        'Frag': frag
    }
    
    return results


def print_results(results: dict):
    """Print results in a formatted table."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-"*60)
    
    for metric, value in results.items():
        if value == -1.0:
            value_str = "N/A (reference set required)"
        else:
            value_str = f"{value:.6f}"
        print(f"{metric:<20} {value_str:<15}")
    
    print("="*60)


def save_results(results: dict, output_file: Optional[str] = None):
    """Save results to a file (supports .txt and .csv formats)."""
    if output_file is None:
        output_file = "evaluation_results.txt"
    
    # Determine format from extension
    if output_file.endswith('.csv'):
        # Save as CSV
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for metric, value in results.items():
                if value == -1.0:
                    value_str = "N/A"
                else:
                    value_str = f"{value:.6f}"
                writer.writerow([metric, value_str])
    else:
        # Save as text file
        with open(output_file, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"{'Metric':<20} {'Value':<15}\n")
            f.write("-"*60 + "\n")
            
            for metric, value in results.items():
                if value == -1.0:
                    value_str = "N/A (reference set required)"
                else:
                    value_str = f"{value:.6f}"
                f.write(f"{metric:<20} {value_str:<15}\n")
            
            f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generated molecules - Optimized Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate without reference set (only Validity, Uniqueness, IntDiv, IntDiv2, Filter)
  python src/evaluate_molecules_optimized.py --generated generated_smiles.txt
  
  # Evaluate with reference set (all metrics, uses MOSES directly if available)
  python src/evaluate_molecules_optimized.py --generated generated_smiles.txt --reference train_smiles.txt
  
  # Use 8 parallel jobs
  python src/evaluate_molecules_optimized.py --generated generated_smiles.txt --reference train_smiles.txt --n_jobs 8
  
  # Disable direct MOSES usage (compute metrics individually)
  python src/evaluate_molecules_optimized.py --generated generated_smiles.txt --reference train_smiles.txt --no-moses-direct
  
  # Save results to CSV
  python src/evaluate_molecules_optimized.py --generated generated_smiles.txt --reference train_smiles.txt --output results.csv
        """
    )
    
    parser.add_argument(
        '--generated',
        type=str,
        required=True,
        help='Path to file containing generated SMILES (one per line)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        default=None,
        help='Path to file containing reference SMILES (training/test set, one per line)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output file for results (default: evaluation_results.txt)'
    )
    
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all CPUs, default: -1)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for SNN computation (default: 1000)'
    )
    
    parser.add_argument(
        '--no-moses-direct',
        action='store_true',
        help='Disable direct use of MOSES metrics (compute individually)'
    )
    
    args = parser.parse_args()
    
    # Check if generated file exists
    if not os.path.exists(args.generated):
        print(f"Error: Generated file not found: {args.generated}")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_molecules(
        args.generated, 
        args.reference,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
        use_moses_direct=not args.no_moses_direct
    )
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output)


if __name__ == '__main__':
    main()

