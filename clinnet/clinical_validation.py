"""
Comprehensive ClinVar Clinical Validation
======================================================
This script performs a detailed clinical validation of CLinNET-predicted genes

"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from collections import Counter, defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# =========================
# CONFIGURATION
# =========================
GENE_FILE = "shap_data_Diag-nervous_system_average.csv"
CLINVAR_FILE = "variant_summary.txt.gz"
OUTPUT_DIR = "clinvar_clinical_validation"
TOP_PERCENTILE = 10  # Top 10% of genes

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1. Load CLinNET genes
# =========================
print("="*70)
print("CLINVAR CLINICAL VALIDATION FOR CLINNET")
print("="*70)

print("\n[1/8] Loading CLinNET genes...")
genes_df = pd.read_csv(GENE_FILE)
genes_df = genes_df.sort_values(by=genes_df.columns[1], ascending=False)

# Calculate top genes
n_top = int(len(genes_df) * (TOP_PERCENTILE / 100))
top_genes_df = genes_df.head(n_top).copy()
top_genes_df['Rank'] = range(1, len(top_genes_df) + 1)

clin_genes = set(top_genes_df["Genes"].astype(str).str.strip().unique())
print(f"   Total genes: {len(genes_df)}")
print(f"   Top {TOP_PERCENTILE}% genes: {len(clin_genes)}")

# =========================
# STEP 2. Load ClinVar data
# =========================
print("\n[2/8] Loading ClinVar dataset...")
clinvar_df = pd.read_csv(
    CLINVAR_FILE, 
    sep="\t", 
    low_memory=False, 
    compression='gzip',
    usecols=['GeneSymbol', 'ClinicalSignificance', 'ReviewStatus', 
             'PhenotypeList', 'Type', 'ClinSigSimple']
)
print(f"   Loaded {len(clinvar_df):,} total ClinVar variants")

# =========================
# STEP 3. Filter for CLinNET genes
# =========================
print("\n[3/8] Filtering for CLinNET genes...")
clinvar_df = clinvar_df[clinvar_df["GeneSymbol"].isin(clin_genes)].copy()
print(f"   Variants in CLinNET genes: {len(clinvar_df):,}")

# =========================
# STEP 4. Clinical significance classification
# =========================
print("\n[4/8] Classifying clinical significance...")

def classify_significance(sig_text):
    """Classify clinical significance"""
    sig_lower = str(sig_text).lower()
    
    if 'pathogenic' in sig_lower and 'conflicting' not in sig_lower:
        if 'likely pathogenic' in sig_lower:
            return 'Likely pathogenic'
        elif 'benign' not in sig_lower:
            return 'Pathogenic'
    elif 'benign' in sig_lower or 'likely benign' in sig_lower:
        return 'Benign/Likely benign'
    elif 'uncertain' in sig_lower or 'vus' in sig_lower:
        return 'VUS'
    
    return 'Other'

clinvar_df['ClinSigCategory'] = clinvar_df['ClinicalSignificance'].apply(classify_significance)

# Filter for pathogenic variants
pathogenic_df = clinvar_df[
    clinvar_df['ClinSigCategory'].isin(['Pathogenic', 'Likely pathogenic'])
].copy()

print(f"   Pathogenic variants: {len(pathogenic_df[pathogenic_df['ClinSigCategory']=='Pathogenic'])}")
print(f"   Likely pathogenic variants: {len(pathogenic_df[pathogenic_df['ClinSigCategory']=='Likely pathogenic'])}")

# =========================
# STEP 5. Evidence quality scoring
# =========================
print("\n[5/8] Assessing evidence quality...")

def score_evidence(review_status):
    """Score evidence quality based on review status"""
    status_lower = str(review_status).lower()
    
    if 'practice guideline' in status_lower:
        return 4, 'Practice guideline'
    elif 'reviewed by expert panel' in status_lower:
        return 3, 'Expert panel'
    elif 'criteria provided, multiple submitters' in status_lower:
        return 2, 'Multiple submitters'
    elif 'criteria provided, single submitter' in status_lower:
        return 1, 'Single submitter'
    else:
        return 0, 'No assertion'

pathogenic_df[['EvidenceScore', 'EvidenceLevel']] = pathogenic_df['ReviewStatus'].apply(
    lambda x: pd.Series(score_evidence(x))
)

print(f"   Evidence quality distribution:")
for level in sorted(pathogenic_df['EvidenceScore'].unique(), reverse=True):
    count = sum(pathogenic_df['EvidenceScore'] == level)
    level_name = pathogenic_df[pathogenic_df['EvidenceScore'] == level]['EvidenceLevel'].iloc[0]
    print(f"     Level {level} ({level_name}): {count} variants")

# =========================
# STEP 6. Phenotype categorization 
# =========================
print("\n[6/8] Categorizing neurological phenotypes...")

phenotype_map = {
    # Core neurodevelopmental
    "neurodevelopment": "Neurodevelopmental disorder",
    "developmental delay": "Neurodevelopmental disorder",
    "psychomotor delay": "Neurodevelopmental disorder",
    
    # Cognitive and intellectual
    "intellectual": "Intellectual disability",
    "cognitive": "Cognitive impairment",
    "learning": "Cognitive impairment",
    "dementia": "Cognitive impairment",
    
    # Autism spectrum and behavior
    "autism": "Autism spectrum disorder",
    "asperger": "Autism spectrum disorder",
    "behavior": "Psychiatric disorder",
    "schizo": "Psychiatric disorder",
    "bipolar": "Psychiatric disorder",
    "depress": "Psychiatric disorder",
    "psychosis": "Psychiatric disorder",
    "adhd": "Psychiatric disorder",
    
    # Seizure and epilepsy
    "epilep": "Epilepsy / seizure disorder",
    "seizure": "Epilepsy / seizure disorder",
    "encephalopathy": "Encephalopathy",
    
    # Neurodegenerative and movement
    "neurodegener": "Neurodegenerative disorder",
    "parkinson": "Movement disorder",
    "ataxia": "Movement disorder",
    "dystonia": "Movement disorder",
    "tremor": "Movement disorder",
    "chorea": "Movement disorder",
    
    # Cerebellar and motor coordination
    "hypotonia": "Motor disorder",
    "motor": "Motor disorder",
    "coordination": "Motor disorder",
    "gait": "Motor disorder",
    
    # Brain structure / malformations
    "microcephaly": "Structural brain abnormality",
    "macrocephaly": "Structural brain abnormality",
    "lissencephaly": "Structural brain abnormality",
    "polymicrogyria": "Structural brain abnormality",
    "cortical dysplasia": "Structural brain abnormality",
    
    # Sensory / communication
    "speech": "Speech/language disorder",
    "language": "Speech/language disorder",
    "hearing": "Sensory disorder",
    "vision": "Sensory disorder",
    "blind": "Sensory disorder",
    "retina": "Sensory disorder",
    "optic": "Sensory disorder",
    
    # Peripheral and muscular
    "neuropathy": "Peripheral neuropathy",
    "myopathy": "Neuromuscular disorder",
    "muscular": "Neuromuscular disorder",
    "als": "Motor neuron disease",
    "spinal": "Spinal cord disorder",
}

def classify_phenotype(phenotype):
    """Classify phenotype into categories"""
    for key, label in phenotype_map.items():
        if re.search(key, str(phenotype), re.IGNORECASE):
            return label
    return "Other/General"

pathogenic_df["NeuroCategory"] = pathogenic_df["PhenotypeList"].apply(classify_phenotype)

# =========================
# STEP 7. Gene-level analysis
# =========================
print("\n[7/8] Analyzing gene-level metrics...")

# Calculate variants per gene
gene_metrics = []

for _, gene_row in top_genes_df.iterrows():
    gene = gene_row['Genes']
    rank = gene_row['Rank']
    importance = gene_row['importance']
    
    gene_variants = pathogenic_df[pathogenic_df['GeneSymbol'] == gene]
    
    path_count = sum(gene_variants['ClinSigCategory'] == 'Pathogenic')
    likely_path_count = sum(gene_variants['ClinSigCategory'] == 'Likely pathogenic')
    max_evidence = gene_variants['EvidenceScore'].max() if len(gene_variants) > 0 else 0
    
    # ND-specific variants
    nd_specific = sum(gene_variants['NeuroCategory'] != 'Other/General')
    
    gene_metrics.append({
        'Gene': gene,
        'Rank': rank,
        'SHAP_Importance': importance,
        'Pathogenic_Variants': path_count,
        'Likely_Pathogenic_Variants': likely_path_count,
        'Total_Pathogenic': path_count + likely_path_count,
        'Max_Evidence_Score': max_evidence,
        'ND_Specific_Variants': nd_specific,
        'Has_Diagnostic_Evidence': (path_count > 0) and (max_evidence >= 2)
    })

gene_metrics_df = pd.DataFrame(gene_metrics)

# Summary statistics
genes_with_variants = sum(gene_metrics_df['Total_Pathogenic'] > 0)
genes_with_high_evidence = sum(gene_metrics_df['Has_Diagnostic_Evidence'])
diagnostic_yield = (genes_with_high_evidence / len(gene_metrics_df)) * 100

print(f"\n   Gene-level summary:")
print(f"     Genes with pathogenic variants: {genes_with_variants}/{len(gene_metrics_df)} ({genes_with_variants/len(gene_metrics_df)*100:.1f}%)")
print(f"     Genes with diagnostic evidence: {genes_with_high_evidence} ({diagnostic_yield:.1f}%)")
print(f"     Total pathogenic variants: {gene_metrics_df['Total_Pathogenic'].sum()}")

# =========================
# STEP 8. Statistical analysis
# =========================
print("\n[8/8] Performing statistical analyses...")

# Correlation between rank and variants
genes_with_vars = gene_metrics_df[gene_metrics_df['Total_Pathogenic'] > 0]
if len(genes_with_vars) > 5:
    corr, p_value = spearmanr(genes_with_vars['Rank'], genes_with_vars['Total_Pathogenic'])
    print(f"\n   Rank-Evidence Correlation:")
    print(f"     Spearman ρ = {corr:.3f} (p = {p_value:.2e})")
else:
    corr, p_value = None, None
    print(f"\n   Rank-Evidence Correlation: Insufficient data")

# Phenotype specificity
nd_specific_count = sum(pathogenic_df['NeuroCategory'] != 'Other/General')
total_count = len(pathogenic_df)
nd_specificity = (nd_specific_count / total_count) * 100

print(f"\n   Phenotype Specificity:")
print(f"     ND-specific variants: {nd_specific_count}/{total_count} ({nd_specificity:.1f}%)")

# Evidence quality in top vs bottom genes
top_20_pct = gene_metrics_df.nsmallest(int(len(gene_metrics_df)*0.2), 'Rank')
bottom_20_pct = gene_metrics_df.nlargest(int(len(gene_metrics_df)*0.2), 'Rank')

top_avg_variants = top_20_pct['Total_Pathogenic'].mean()
bottom_avg_variants = bottom_20_pct['Total_Pathogenic'].mean()

print(f"\n   Rank-based comparison:")
print(f"     Top 20% genes: {top_avg_variants:.2f} variants/gene (avg)")
print(f"     Bottom 20% genes: {bottom_avg_variants:.2f} variants/gene (avg)")

# =========================
# EXPORT RESULTS
# =========================
print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

# 1. Gene-level summary
gene_metrics_df.to_csv(f"{OUTPUT_DIR}/gene_clinical_summary.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/gene_clinical_summary.csv")

# 2. Phenotype breakdown
category_counts = pathogenic_df.groupby('NeuroCategory').agg({
    'GeneSymbol': 'nunique',
    'Type': 'count'
}).reset_index()
category_counts.columns = ['Phenotype_Category', 'Unique_Genes', 'Total_Variants']
category_counts = category_counts.sort_values('Unique_Genes', ascending=False)
category_counts.to_csv(f"{OUTPUT_DIR}/phenotype_breakdown.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/phenotype_breakdown.csv")

# 3. Evidence quality summary
evidence_summary = pathogenic_df.groupby(['ClinSigCategory', 'EvidenceLevel']).size().reset_index()
evidence_summary.columns = ['Clinical_Significance', 'Evidence_Level', 'Variant_Count']
evidence_summary.to_csv(f"{OUTPUT_DIR}/evidence_quality_summary.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/evidence_quality_summary.csv")

# 4. Top validated genes
top_validated = gene_metrics_df.nlargest(30, 'Total_Pathogenic')[
    ['Gene', 'Rank', 'SHAP_Importance', 'Pathogenic_Variants', 
     'Likely_Pathogenic_Variants', 'Max_Evidence_Score', 'ND_Specific_Variants']
]
top_validated.to_csv(f"{OUTPUT_DIR}/top_validated_genes.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/top_validated_genes.csv")

# 5. Summary statistics
summary_stats = pd.DataFrame({
    'Metric': [
        'Total Top Genes Analyzed',
        'Genes with Pathogenic Variants',
        'Percentage with Variants',
        'Genes with Diagnostic Evidence (High Quality)',
        'Diagnostic Yield (%)',
        'Total Pathogenic Variants',
        'ND-Specific Variants',
        'ND Specificity (%)',
        'Rank-Evidence Correlation (Spearman ρ)',
        'Correlation P-value'
    ],
    'Value': [
        len(gene_metrics_df),
        genes_with_variants,
        f"{genes_with_variants/len(gene_metrics_df)*100:.2f}%",
        genes_with_high_evidence,
        f"{diagnostic_yield:.2f}%",
        gene_metrics_df['Total_Pathogenic'].sum(),
        nd_specific_count,
        f"{nd_specificity:.1f}%",
        f"{corr:.3f}" if corr else "N/A",
        f"{p_value:.2e}" if p_value else "N/A"
    ]
})
summary_stats.to_csv(f"{OUTPUT_DIR}/summary_statistics.csv", index=False)
print(f"✓ Saved: {OUTPUT_DIR}/summary_statistics.csv")

print("\n" + "="*70)
print("DATA ANALYSIS COMPLETE - GENERATING FIGURES")
print("="*70)

# =========================
# FIGURE FUNCTIONS
# =========================

def figure1_phenotype_breakdown():
    """Figure 1: Phenotype category breakdown (Advanced Science style, pastel-aligned)"""
    print("\n[Figure 1] Generating phenotype breakdown...")

    # --- Font and figure style ---
    rcParams['font.family'] = 'Times New Roman'
    rcParams['axes.linewidth'] = 1.2
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12

    # --- Prepare data ---
    category_data = (
        pathogenic_df.groupby('NeuroCategory')
        .agg({'GeneSymbol': 'nunique', 'Type': 'count'})
        .reset_index()
    )
    category_data.columns = ['Category', 'Unique_Genes', 'Total_Variants']
    category_data = category_data.sort_values('Unique_Genes', ascending=False)

    # Optionally remove "Other/General"
    category_data = category_data[category_data['Category'] != 'Other/General']

    # --- Pastel color palette (consistent with Figure 7 and Figure 2) ---
    pastel_palette = ['#f9c5d1', '#c9f7c0', '#b3e5fc', '#b8c0ff', '#e2afff']
    colors = pastel_palette * (len(category_data) // len(pastel_palette) + 1)
    colors = colors[:len(category_data)]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 7))

    bars = ax.barh(
        range(len(category_data)),
        category_data['Unique_Genes'],
        color=colors,
        edgecolor='white',
        linewidth=1.3,
        alpha=0.95
    )

    # --- Axis labels and title ---
    ax.set_yticks(range(len(category_data)))
    ax.set_yticklabels(category_data['Category'], fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Genes with Pathogenic Variants',
                  fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Clinical Phenotype Category',
                  fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title('ClinVar Clinical Evidence by Phenotype Category',
                 fontsize=17, fontweight='bold', pad=20)

    # --- Value labels ---
    for i, (bar, val) in enumerate(zip(bars, category_data['Unique_Genes'])):
        ax.text(
            bar.get_width() + max(category_data['Unique_Genes']) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val}",
            va='center',
            fontsize=13,
            fontweight='bold',
            color='black'
        )

    # --- Styling ---
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # --- Save ---
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure1_Phenotype_Breakdown.png",
                dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure1_Phenotype_Breakdown.png")


def figure2_evidence_quality():
    """Figure 2: Evidence quality pyramid (Advanced Science style, pastel-aligned)"""
    print("\n[Figure 2] Generating evidence quality pyramid...")

    # --- Font and figure style ---
    rcParams['font.family'] = 'Times New Roman'
    rcParams['axes.linewidth'] = 1.2
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12

    # --- Prepare data ---
    evidence_counts = (
        pathogenic_df.groupby(['EvidenceScore', 'EvidenceLevel'])
        .size()
        .reset_index()
    )
    evidence_counts.columns = ['Score', 'Level', 'Count']
    evidence_counts = evidence_counts.sort_values('Score', ascending=True)

    # --- Pastel palette aligned with Figure 7 ---
    pastel_colors = ['#f9c5d1', '#c9f7c0', '#b3e5fc', '#b8c0ff', '#e2afff']
    colors = pastel_colors[:len(evidence_counts)]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # Horizontal bar chart
    bars = ax.barh(
        range(len(evidence_counts)),
        evidence_counts['Count'],
        color=colors,
        edgecolor='white',
        linewidth=1.3,
        alpha=0.95
    )

    # --- Axis labels and title ---
    ax.set_yticks(range(len(evidence_counts)))
    ax.set_yticklabels(
        [f"{row['Level']}\n(n={row['Count']})" for _, row in evidence_counts.iterrows()],
        fontsize=13,
        fontweight='bold'
    )
    ax.set_xlabel('Number of Pathogenic Variants', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title(
        'Clinical Evidence Quality Hierarchy',
        fontsize=17,
        fontweight='bold',
        pad=20
    )

    # --- Value labels ---
    for i, (bar, count) in enumerate(zip(bars, evidence_counts['Count'])):
        ax.text(
            bar.get_width() + max(evidence_counts['Count']) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va='center',
            fontsize=13,
            fontweight='bold',
            color='black'
        )

    # --- Style adjustments ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # --- Save ---
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure2_Evidence_Quality.png",
                dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure2_Evidence_Quality.png")



def figure3_variant_burden_by_rank():
    """Figure 3: Variant burden across gene rank bins"""
    print("\n[Figure 3] Generating variant burden by rank...")
    
    # Create rank bins
    n_bins = 5
    genes_per_bin = len(gene_metrics_df) // n_bins
    
    bin_data = []
    bin_labels = []
    
    for i in range(n_bins):
        start_idx = i * genes_per_bin
        end_idx = (i + 1) * genes_per_bin if i < n_bins - 1 else len(gene_metrics_df)
        
        bin_genes = gene_metrics_df.iloc[start_idx:end_idx]
        variant_counts = bin_genes[bin_genes['Total_Pathogenic'] > 0]['Total_Pathogenic'].tolist()
        
        if not variant_counts:
            variant_counts = [0]
        
        bin_data.append(variant_counts)
        
        start_pct = (i * 100 // n_bins)
        end_pct = ((i + 1) * 100 // n_bins)
        bin_labels.append(f"{start_pct}-{end_pct}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create violin plot
    parts = ax.violinplot(bin_data, positions=range(n_bins),
                         showmeans=True, showmedians=True, widths=0.7)
    
    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(sns.color_palette("Blues_r", n_bins)[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
    
    # Customize other elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.5)
    
    # Customize axes
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_xlabel('Gene Rank Percentile (CLinNet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pathogenic Variants per Gene', fontsize=12, fontweight='bold')
    ax.set_title('Variant Burden Distribution Across Gene Rankings', 
                fontsize=14, fontweight='bold', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure3_Variant_Burden_By_Rank.png", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure3_Variant_Burden_By_Rank.png")


def figure4_rank_correlation():
    """Figure 4: Gene rank vs pathogenic variant count correlation"""
    print("\n[Figure 4] Generating rank-evidence correlation...")
    
    # Filter genes with variants
    plot_data = gene_metrics_df[gene_metrics_df['Total_Pathogenic'] > 0].copy()
    
    if len(plot_data) < 5:
        print("   ⚠ Insufficient data for correlation plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    scatter = ax.scatter(plot_data['Rank'], 
                        plot_data['Total_Pathogenic'],
                        s=80, 
                        c=plot_data['SHAP_Importance'],
                        cmap='viridis',
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5)
    
    # Add trend line
    if len(plot_data) > 2:
        z = np.polyfit(plot_data['Rank'], plot_data['Total_Pathogenic'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data['Rank'].min(), plot_data['Rank'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Trend line')
    
    # Add correlation text
    if corr and p_value:
        ax.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_value:.2e}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='SHAP Importance Score')
    cbar.ax.tick_params(labelsize=10)
    
    # Customize
    ax.set_xlabel('Gene Rank (CLinNet)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Pathogenic Variants', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Between Gene Rank and Clinical Evidence', 
                fontsize=14, fontweight='bold', pad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    
    if len(plot_data) > 2:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure4_Rank_Correlation.png", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure4_Rank_Correlation.png")



from matplotlib import rcParams

def figure5_diagnostic_yield():
    """Figure 5: Cumulative diagnostic yield curve (Advanced Science style)"""
    print("\n[Figure 5] Generating diagnostic yield curve...")

    # --- Font and style settings ---
    #rcParams['font.family'] = 'Times New Roman'
    rcParams['axes.linewidth'] = 1.2
    #rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12

    # --- Prepare data ---
    diagnostic_genes = gene_metrics_df[gene_metrics_df['Has_Diagnostic_Evidence']].copy()
    diagnostic_genes = diagnostic_genes.sort_values('Rank')

    if len(diagnostic_genes) == 0:
        print("   ⚠ No diagnostic genes to plot")
        return

    ranks = list(range(1, len(gene_metrics_df) + 1))
    cumulative = [sum(diagnostic_genes['Rank'] <= r) for r in ranks]
    percentiles = np.array(ranks) / len(gene_metrics_df) * 100

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Main cumulative curve ---
    ax.plot(percentiles, cumulative, linewidth=3.2, color='#264653',
            label='Cumulative Diagnostic Yield')
    ax.fill_between(percentiles, cumulative, alpha=0.25, color='#2a9d8f')

    # --- 50% diagnostic marker ---
    half_diagnostic = len(diagnostic_genes) / 2
    if max(cumulative) > half_diagnostic:
        half_idx = next(i for i, v in enumerate(cumulative) if v >= half_diagnostic)
        half_percentile = percentiles[half_idx]
        ax.axhline(half_diagnostic, color='#e63946', linestyle='--', linewidth=1.8)
        ax.axvline(half_percentile, color='#e63946', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(half_percentile + 2, half_diagnostic,
                f'50% yield at {half_percentile:.1f}%',
                fontsize=13, fontweight='bold', color='#e63946')

    # --- Axis labels and title ---
    ax.set_xlabel('Top Gene Percentile', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cumulative Diagnostic Genes', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title('Cumulative Diagnostic Yield by Gene Ranking',
                 fontsize=17, fontweight='bold', pad=20)

    # --- Clean presentation ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=13, frameon=False)

    # --- Save figure ---
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure5_Diagnostic_Yield.png",
                dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure5_Diagnostic_Yield.png")


def figure6_variant_types():
    """Figure 6: Variant type distribution"""
    print("\n[Figure 6] Generating variant type distribution...")
    
    # Count variant types
    type_counts = pathogenic_df['Type'].value_counts().head(8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create pie chart
    colors = sns.color_palette("Set2", len(type_counts))
    wedges, texts, autotexts = ax.pie(type_counts.values, 
                                       labels=type_counts.index,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors,
                                       textprops={'fontsize': 11})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Make labels bold
    for text in texts:
        text.set_fontweight('bold')
        text.set_fontsize(11)
    
    ax.set_title('Distribution of Pathogenic Variant Types', 
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure6_Variant_Types.png", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure6_Variant_Types.png")


from matplotlib.colors import LinearSegmentedColormap

def figure7_top_genes_heatmap():
    """Figure 7: Top 20 genes clinical profile heatmap (vertical, pastel-blended style)"""
    print("\n[Figure 7] Generating top genes heatmap...")

    # Get top 20 genes by total variants
    top20 = gene_metrics_df.nlargest(20, 'Total_Pathogenic')

    if len(top20) == 0:
        print("   ⚠ No genes with variants for heatmap")
        return

    # Prepare data
    matrix = top20[['Pathogenic_Variants', 'Likely_Pathogenic_Variants',
                    'Max_Evidence_Score', 'ND_Specific_Variants']].T
    matrix.columns = top20['Gene']

    # Custom smooth pastel gradient
    pastel_blend = LinearSegmentedColormap.from_list(
        "pastel_blend",
        ['#f9c5d1', '#c9f7c0', '#b3e5fc'],  # soft pink → light green → sky blue
        N=256
    )

    # Create figure
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, cmap=pastel_blend, annot=True, fmt='.0f',
                     linewidths=0.5, linecolor='white', square=False,
                     cbar_kws={'label': 'Value'},
                     annot_kws={'fontsize': 8,
                                #'fontweight': 'bold', 
                                'color': 'black'})

    # Axis labels and ticks
    ax.set_xlabel('Genes', fontsize=12, fontweight='bold', labelpad=5)
    ax.set_ylabel('Clinical Evidence Metrics', fontsize=12, fontweight='bold', labelpad=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9, ha='center')
    ax.set_yticklabels(['Pathogenic\nVariants', 'Likely Pathogenic\nVariants',
                        'Max Evidence\nScore', 'ND-Specific\nVariants'],
                       fontsize=10, fontweight='medium', rotation=0)

    # Title
   # ax.set_title('Clinical Evidence Profile: Top 20 Genes',
    #             fontsize=14,  pad=10)

    # Style adjustments for publication
    ax.figure.patch.set_alpha(0.0)  # transparent background
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Value', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure7_Top_Genes_Heatmap.png",
                dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure7_Top_Genes_Heatmap.png")



def figure8_nd_specificity():
    """Figure 8: ND-specificity comparison"""
    print("\n[Figure 8] Generating ND-specificity comparison...")
    
    # Count ND-specific vs general
    nd_specific = pathogenic_df[pathogenic_df['NeuroCategory'] != 'Other/General']
    general = pathogenic_df[pathogenic_df['NeuroCategory'] == 'Other/General']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    categories = ['ND-Specific\nPhenotypes', 'General\nDisease']
    counts = [len(nd_specific), len(general)]
    colors_bar = ['#e74c3c', '#95a5a6']
    
    bars = ax.bar(categories, counts, color=colors_bar, 
                 edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value labels and percentages
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize
    ax.set_ylabel('Number of Pathogenic Variants', fontsize=12, fontweight='bold')
    ax.set_title('Phenotype Specificity of CLinNet-Identified Genes', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, max(counts) * 1.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Figure8_ND_Specificity.png", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {OUTPUT_DIR}/Figure8_ND_Specificity.png")


# =========================
# GENERATE ALL FIGURES
# =========================
print("\n" + "="*70)
print("GENERATING ALL FIGURES")
print("="*70)

figure1_phenotype_breakdown()
figure2_evidence_quality()
figure3_variant_burden_by_rank()
figure4_rank_correlation()
figure5_diagnostic_yield()
figure6_variant_types()
figure7_top_genes_heatmap()
figure8_nd_specificity()

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

print(f"\n📊 KEY FINDINGS:")
print(f"   • {genes_with_variants}/{len(gene_metrics_df)} genes ({genes_with_variants/len(gene_metrics_df)*100:.1f}%) with pathogenic variants")
print(f"   • {diagnostic_yield:.1f}% diagnostic yield (high-quality evidence)")
print(f"   • {gene_metrics_df['Total_Pathogenic'].sum()} total pathogenic variants")
print(f"   • {nd_specificity:.1f}% ND-specific variants")
if corr and p_value:
    print(f"   • Rank-evidence correlation: ρ = {corr:.3f} (p = {p_value:.2e})")

print(f"\n📁 OUTPUT FILES:")
print(f"   • {OUTPUT_DIR}/gene_clinical_summary.csv")
print(f"   • {OUTPUT_DIR}/phenotype_breakdown.csv")
print(f"   • {OUTPUT_DIR}/evidence_quality_summary.csv")
print(f"   • {OUTPUT_DIR}/top_validated_genes.csv")
print(f"   • {OUTPUT_DIR}/summary_statistics.csv")

print(f"\n🎨 FIGURES GENERATED:")
for i in range(1, 9):
    print(f"   • Figure {i}: {OUTPUT_DIR}/Figure{i}_*.png")

print("\n" + "="*70)
print("✅ ALL ANALYSES COMPLETED SUCCESSFULLY!")
print("="*70)
