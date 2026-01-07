"""
Evaluation script for generated molecules.
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

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from moses.metrics import get_all_metrics
    MOSES_AVAILABLE = True
except ImportError:
    MOSES_AVAILABLE = False
    print("Warning: MOSES library not available. Filter and SNN metrics will use alternative implementations.")

try:
    from fcd import get_fcd
    FCD_AVAILABLE = True
except ImportError:
    FCD_AVAILABLE = False
    print("Warning: FCD library not available. FCD metric will be skipped. Install with: pip install fcd-torch")


def load_smiles_from_file(filepath: str) -> List[str]:
    """Load SMILES strings from a text file (one per line)."""
    smiles_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.lower() != 'none':
                smiles_list.append(line)
    return smiles_list


def compute_validity(smiles_list: List[str]) -> Tuple[float, List[str]]:
    """Compute validity: fraction of valid SMILES."""
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_smiles.append(smiles)
            except:
                pass
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


def compute_scaffold_similarity(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> float:
    """Compute scaffold similarity: fraction of generated scaffolds present in reference set."""
    if reference_smiles is None:
        return -1.0
    
    # Get scaffolds from generated molecules
    generated_scaffolds = set()
    for smiles in unique_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                generated_scaffolds.add(scaffold_smiles)
        except:
            pass
    
    # Get scaffolds from reference molecules
    reference_scaffolds = set()
    for smiles in reference_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                reference_scaffolds.add(scaffold_smiles)
        except:
            pass
    
    # Compute intersection
    common_scaffolds = generated_scaffolds.intersection(reference_scaffolds)
    scaff = len(common_scaffolds) / len(generated_scaffolds) if len(generated_scaffolds) > 0 else 0.0
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


def compute_filter(unique_smiles: List[str]) -> float:
    """Compute Filter: fraction of molecules passing MOSES filters."""
    if not MOSES_AVAILABLE:
        # Alternative implementation using basic filters
        passed = 0
        for smiles in unique_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Basic filters: no unusual valences, sanitizable
                    Chem.SanitizeMol(mol)
                    # Check for PAINS and other problematic substructures
                    if not has_pains(mol):
                        passed += 1
            except:
                pass
        return passed / len(unique_smiles) if len(unique_smiles) > 0 else 0.0
    
    try:
        metrics = get_all_metrics(unique_smiles, train=None, device='cpu')
        return metrics.get('Filters', -1.0)
    except Exception as e:
        print(f"Error computing Filter: {e}")
        return -1.0


def has_pains(mol) -> bool:
    """Check for PAINS (Pan Assay Interference Compounds) patterns."""
    # Simplified PAINS check - in practice, use a comprehensive PAINS filter
    pains_patterns = [
        '[N+](=O)[O-]',  # Nitro group
        '[S,P](=O)(=O)',  # Sulfonyl/Phosphonyl
    ]
    for pattern in pains_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return True
    return False


def compute_snn(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> float:
    """Compute SNN (Similarity to Nearest Neighbor)."""
    if reference_smiles is None:
        return -1.0
    
    if not MOSES_AVAILABLE:
        # Alternative implementation using Tanimoto similarity
        # Pre-compute reference fingerprints for efficiency
        ref_fps = []
        for ref_smiles in reference_smiles:
            try:
                ref_mol = Chem.MolFromSmiles(ref_smiles)
                if ref_mol is not None:
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                    ref_fps.append(ref_fp)
            except:
                continue
        
        if len(ref_fps) == 0:
            return -1.0
        
        similarities = []
        for gen_smiles in unique_smiles:
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
        
        return np.mean(similarities) if len(similarities) > 0 else -1.0
    
    try:
        metrics = get_all_metrics(unique_smiles, train=reference_smiles, device='cpu')
        return metrics.get('SNN', -1.0)
    except Exception as e:
        print(f"Error computing SNN: {e}")
        return -1.0


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
        indices = list(range(n_fps))
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
    mean_sim = np.mean(similarities)
    if p == 1:
        return 1.0 - mean_sim
    elif p == 2:
        mean_sim_sq = np.mean([s**2 for s in similarities])
        return 1.0 - np.sqrt(mean_sim_sq)
    else:
        raise ValueError(f"Invalid p value: {p}. Must be 1 or 2.")


def compute_fragment_similarity(unique_smiles: List[str], reference_smiles: Optional[List[str]] = None) -> float:
    """Compute Fragment Similarity: similarity of fragment distributions."""
    if reference_smiles is None:
        return -1.0
    
    def get_fragments(smiles_list):
        """Extract molecular fragments (BRICS fragments)."""
        fragments = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Use BRICS to break into fragments
                    frags = Chem.rdmolops.BRICSDecompose(mol)
                    fragments.extend([Chem.MolToSmiles(f) for f in frags])
            except:
                continue
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


def evaluate_molecules(generated_file: str, reference_file: Optional[str] = None) -> dict:
    """
    Evaluate generated molecules against reference set.
    
    Args:
        generated_file: Path to file with generated SMILES (one per line)
        reference_file: Optional path to file with reference SMILES (for Novelty, Scaff, FCD, SNN, Frag)
    
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
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # Validity
    print("  Computing Validity...")
    validity, valid_smiles = compute_validity(generated_smiles)
    
    # Uniqueness
    print("  Computing Uniqueness...")
    uniqueness, unique_smiles = compute_uniqueness(valid_smiles)
    
    # Novelty
    print("  Computing Novelty...")
    novelty = compute_novelty(unique_smiles, reference_smiles)
    
    # Scaff
    print("  Computing Scaff...")
    scaff = compute_scaffold_similarity(unique_smiles, reference_smiles)
    
    # FCD
    print("  Computing FCD...")
    fcd = compute_fcd(unique_smiles, reference_smiles)
    
    # Filter
    print("  Computing Filter...")
    filter_metric = compute_filter(unique_smiles)
    
    # SNN
    print("  Computing SNN...")
    snn = compute_snn(unique_smiles, reference_smiles)
    
    # IntDiv
    print("  Computing IntDiv...")
    intdiv = compute_internal_diversity(unique_smiles, p=1)
    
    # IntDiv2
    print("  Computing IntDiv2...")
    intdiv2 = compute_internal_diversity(unique_smiles, p=2)
    
    # Frag
    print("  Computing Frag...")
    frag = compute_fragment_similarity(unique_smiles, reference_smiles)
    
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
        description='Evaluate generated molecules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate without reference set (only Validity, Uniqueness, IntDiv, IntDiv2, Filter)
  python src/evaluate_molecules.py --generated generated_samples/generated_smiles.txt
  
  # Evaluate with reference set (all metrics)
  python src/evaluate_molecules.py --generated generated_samples/generated_smiles.txt --reference data/train_smiles.txt
  
  # Save results to file
  python src/evaluate_molecules.py --generated generated_samples/generated_smiles.txt --reference data/train_smiles.txt --output results.txt
  
  # Save results as CSV
  python src/evaluate_molecules.py --generated generated_samples/generated_smiles.txt --reference data/train_smiles.txt --output results.csv
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
    
    args = parser.parse_args()
    
    # Check if generated file exists
    if not os.path.exists(args.generated):
        print(f"Error: Generated file not found: {args.generated}")
        sys.exit(1)
    
    # Evaluate
    results = evaluate_molecules(args.generated, args.reference)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output)


if __name__ == '__main__':
    main()

