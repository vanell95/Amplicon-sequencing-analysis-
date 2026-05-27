# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:12:56 2026

@author: Anelli
"""

# -*- coding: utf-8 -*-
"""
STEP 1 of Naples dataset 16S analysis: Filtering and per-sample summary

Goal:
- Read the original ZOTU table
- Keep Bacteria AND Archaea (Archaea included because marine archaeal lineages
  such as Thaumarchaeota are motile and chemotaxis-capable)
- Remove chloroplast and mitochondria
- Remove putative contaminant genera identified from controls
- Keep only BULK, FSW, and 2216 samples
- Compute a per-sample overview table (total reads, observed ZOTUs, Shannon diversity)

Input:
- p1112_run260227_16S_ZOTU_Count_SILVA138.txt  (raw MiSeq ZOTU table)

Output:
1. filtered_zotu_table_bacteria_no_contaminants.txt
2. sample_summary_filtered.csv
3. sample_metadata_filtered.csv
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE = Path("C:/Users/Anelli/Desktop/Experiments/GDC_sequencing/Naples_dataset/p1112_run260227_16S_Results/e_OTU/p1112_run260227_16S_ZOTU_Count_SILVA138.txt")

OUTPUT_FILTERED_TABLE = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_SUMMARY = Path("sample_summary_filtered.csv")
OUTPUT_METADATA = Path("sample_metadata_filtered.csv")

GROUPS_TO_KEEP = ["BULK", "FSW", "2216"]

# contaminant genera identified from undeployed FSW controls
CONTAMINANT_GENERA = {
    "Bradyrhizobium",
    "Methylobacterium",
    "Methylorubrum",
    "Cutibacterium",
    "Escherichia-Shigella",
    "Enhydrobacter",
    "Paracoccus"
}

# =========================
# 3. READ INPUT TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col = "#OTU ID"
taxonomy_col = "Consensus Lineage"

sample_cols = [col for col in df.columns if col not in [zotu_col, taxonomy_col]]

print("Original table loaded.")
print("Shape:", df.shape)
print("Number of samples:", len(sample_cols))

# =========================
# 4. ASSIGN SAMPLES TO GROUPS
# =========================
def assign_group(sample_name):
    """
    Assign each sample to a biological group based on its name.
    """
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    elif sample_name.startswith("CTR-E-"):
        return "CTR-E"
    elif sample_name.startswith("CTR-PCR"):
        return "CTR-PCR"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

print("\nAll detected samples:")
print(sample_metadata)

# keep only the three biological groups of interest
sample_metadata_filtered = sample_metadata[
    sample_metadata["Group"].isin(GROUPS_TO_KEEP)
].copy()

print("\nSamples kept for analysis:")
print(sample_metadata_filtered)

# save metadata
sample_metadata_filtered.to_csv(OUTPUT_METADATA, index=False)

# get the actual sample names we will keep
samples_to_keep = sample_metadata_filtered["Sample"].tolist()

# =========================
# 5. FILTER 1:
# KEEP BACTERIA AND ARCHAEA
# =========================
# Archaea are included because several marine archaeal lineages
# (e.g. Thaumarchaeota / Nitrososphaeria, including Candidatus
# Nitrosopelagicus) are motile and chemotaxis-capable. Excluding
# them would be methodologically inconsistent with the biological
# question of which microorganisms respond to the chemoattractant.
#
# Chloroplast and mitochondrial reads are still removed in Filter 2
# as these are non-target organelle sequences.
# Contaminant bacterial genera are still removed in Filter 3.

def is_prokaryotic_read(tax_string):
    """
    Keep rows classified as Bacteria OR Archaea.
    Exclude Eukaryota and unassigned sequences.
    """
    if pd.isna(tax_string):
        return False
    t = str(tax_string)
    return ("d__Bacteria" in t) or ("d__Archaea" in t)

n_before = len(df)
df = df[df[taxonomy_col].apply(is_prokaryotic_read)].copy()
n_after_bacteria = len(df)

print("\nAfter keeping Bacteria and Archaea:")
print(f"Rows before: {n_before}")
print(f"Rows after : {n_after_bacteria}")

# How many are Archaea?
n_archaea = df[df[taxonomy_col].apply(lambda t: "d__Archaea" in str(t))].shape[0]
n_bacteria = df[df[taxonomy_col].apply(lambda t: "d__Bacteria" in str(t))].shape[0]
print(f"  of which Bacteria : {n_bacteria}")
print(f"  of which Archaea  : {n_archaea}")

# =========================
# 6. FILTER 2:
# REMOVE CHLOROPLAST AND MITOCHONDRIA
# =========================
def is_not_organelle(tax_string):
    """
    Remove chloroplast and mitochondrial reads.
    """
    if pd.isna(tax_string):
        return False

    tax_lower = str(tax_string).lower()
    return ("chloroplast" not in tax_lower) and ("mitochondria" not in tax_lower)

n_before_organelle = len(df)
df = df[df[taxonomy_col].apply(is_not_organelle)].copy()
n_after_organelle = len(df)

print("\nAfter removing chloroplast and mitochondria:")
print(f"Rows before: {n_before_organelle}")
print(f"Rows after : {n_after_organelle}")

# =========================
# 7. EXTRACT GENUS
# =========================
def extract_genus(tax_string):
    """
    Extract genus from the SILVA taxonomy string.
    If genus is missing, fall back to family or order.
    """
    if pd.isna(tax_string):
        return "Unassigned"

    parts = [p.strip() for p in str(tax_string).split(";") if p.strip()]

    # Try genus
    for part in parts:
        if part.startswith("g__"):
            genus = part.replace("g__", "").strip()
            if genus and genus.lower() not in ["uncultured", "unclassified", "unknown", "metagenome"]:
                return genus

    # Fallback to family
    for part in parts:
        if part.startswith("f__"):
            family = part.replace("f__", "").strip()
            if family:
                return f"{family} (family)"

    # Fallback to order
    for part in parts:
        if part.startswith("o__"):
            order = part.replace("o__", "").strip()
            if order:
                return f"{order} (order)"

    return "Unassigned"

df["Genus"] = df[taxonomy_col].apply(extract_genus)

print("\nExample taxonomy-to-genus assignments:")
print(df[[taxonomy_col, "Genus"]].head(10))

# =========================
# 8. FILTER 3:
# REMOVE PUTATIVE CONTAMINANT GENERA
# =========================
n_before_contaminants = len(df)
df = df[~df["Genus"].isin(CONTAMINANT_GENERA)].copy()
n_after_contaminants = len(df)

print("\nAfter removing putative contaminant genera:")
print("Removed genera:", sorted(CONTAMINANT_GENERA))
print(f"Rows before: {n_before_contaminants}")
print(f"Rows after : {n_after_contaminants}")

# =========================
# 9. KEEP ONLY RELEVANT SAMPLE COLUMNS
# =========================
filtered_df = df[[zotu_col, taxonomy_col, "Genus"] + samples_to_keep].copy()

print("\nFiltered table shape:")
print(filtered_df.shape)

# save filtered table
filtered_df.to_csv(OUTPUT_FILTERED_TABLE, sep="\t", index=False)
print(f"\nSaved filtered table to: {OUTPUT_FILTERED_TABLE}")

# =========================
# 10. DEFINE SHANNON DIVERSITY FUNCTION
# =========================
def shannon_diversity(counts):
    """
    Compute Shannon diversity from a vector of counts.
    """
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]

    if len(counts) == 0:
        return 0.0

    proportions = counts / counts.sum()
    return entropy(proportions)

# =========================
# 11. COMPUTE PER-SAMPLE SUMMARY
# =========================
summary_rows = []

for sample in samples_to_keep:
    counts = filtered_df[sample]

    total_reads = counts.sum()
    observed_zotus = (counts > 0).sum()
    shannon = shannon_diversity(counts)

    summary_rows.append({
        "Sample": sample,
        "Group": sample_metadata_filtered.loc[
            sample_metadata_filtered["Sample"] == sample, "Group"
        ].values[0],
        "Total_reads": int(total_reads),
        "Observed_ZOTUs": int(observed_zotus),
        "Shannon_diversity": round(shannon, 3)
    })

summary_df = pd.DataFrame(summary_rows)

# sort by group then sample
group_order = ["BULK", "FSW", "2216"]
summary_df["Group"] = pd.Categorical(summary_df["Group"], categories=group_order, ordered=True)
summary_df = summary_df.sort_values(["Group", "Sample"]).reset_index(drop=True)

print("\nPer-sample summary:")
print(summary_df)

# save summary
summary_df.to_csv(OUTPUT_SUMMARY, index=False)
print(f"\nSaved per-sample summary to: {OUTPUT_SUMMARY}")

# =========================
# 12. OPTIONAL: GROUP-LEVEL SUMMARY
# =========================
group_summary = (
    summary_df
    .groupby("Group", observed=False)
    .agg(
        n_samples=("Sample", "count"),
        mean_reads=("Total_reads", "mean"),
        mean_observed_ZOTUs=("Observed_ZOTUs", "mean"),
        mean_shannon=("Shannon_diversity", "mean")
    )
    .reset_index()
)

print("\nGroup-level overview:")
print(group_summary)
#%%
# =========================
# 13. PLOT SAMPLE OVERVIEW
# =========================
"""
STEP 2 of Naples dataset 16S analysis: Per-sample overview plots

Goal:
- Visualize sequencing depth, observed ZOTUs, and Shannon diversity for each sample
- Confirm that all three groups (BULK, FSW, 2216) yielded sufficient bacterial
  communities for downstream comparison

Input:
- summary_df  (computed in Step 1, available in memory)

Output:
- overview_total_reads.png
- overview_observed_zotus.png
- overview_shannon.png
"""
import matplotlib.pyplot as plt

# Make sure samples appear in a logical order
sample_order = [
    "BULK-1D", "BULK-2D", "BULK-3D",
    "FSW-1D", "FSW-2D", "FSW-3D", "FSW-4D",
    "2216-1D", "2216-2D", "2216-3D", "2216-4D"
]

summary_df["Sample"] = pd.Categorical(
    summary_df["Sample"],
    categories=sample_order,
    ordered=True
)

summary_df = summary_df.sort_values("Sample").reset_index(drop=True)

# -------- Plot 1: total reads --------
group_colors = {
    "BULK": "#1f77b4",
    "FSW": "#ff7f0e",
    "2216": "#2ca02c"
}

bar_colors = summary_df["Group"].map(group_colors)

fig, ax = plt.subplots(figsize=(20, 14))
ax.bar(summary_df["Sample"], summary_df["Total_reads"], color = bar_colors)
ax.set_ylabel("Total reads", size = 30)
ax.set_title("Sequencing depth per sample", size = 30)
plt.xticks(rotation=45, ha="right", size = 28)
plt.yscale('symlog')
plt.ylim(0,1000000)
plt.yticks(size = 28)
plt.tight_layout()
plt.savefig("overview_total_reads.png", dpi=300, bbox_inches="tight")
plt.show()

# -------- Plot 2: observed ZOTUs --------
fig, ax = plt.subplots(figsize=(20,14))
ax.bar(summary_df["Sample"], summary_df["Observed_ZOTUs"], color = bar_colors)
ax.set_ylabel("Observed ZOTUs", size = 30)
ax.set_title("Observed ZOTUs per sample", size = 30)
plt.xticks(rotation=45, ha="right", size = 28)
plt.yticks(size = 28)
plt.tight_layout()
plt.savefig("overview_observed_zotus.png", dpi=300, bbox_inches="tight")
plt.show()

# -------- Plot 3: Shannon diversity --------
fig, ax = plt.subplots(figsize=(20,14))
ax.bar(summary_df["Sample"], summary_df["Shannon_diversity"], color = bar_colors)
ax.set_ylabel("Shannon diversity Index", size = 30)
ax.set_title("Shannon diversity per sample", size = 30)
plt.xticks(rotation=45, ha="right", size = 28)
plt.yticks(size = 28)
plt.tight_layout()
plt.savefig("overview_shannon.png", dpi=300, bbox_inches="tight")
plt.show()

# -------- Plot 4: Alpha-diversity comparison between groups --------
# ---------------------------------------------------------------
# BULK is shown as a lighter background reference (reduced opacity)
# to provide ecological context without being the focus of the
# primary statistical comparison.
#
# The key comparison is FSW vs 2216:
#   FSW  = passive entry (no chemoattractant)
#   2216 = chemotaxis-driven enrichment
#
# BULK = ambient community reference — kept for context but
#        visually de-emphasised so it does not distract from
#        the main FSW vs 2216 story.
#
# STATISTICAL APPROACH — Welch's t-test
# ---------------------------------------------------------------
# We use Welch's t-test (not Mann-Whitney or Kruskal-Wallis) because:
#
# 1. Shannon diversity is a continuous index on a narrow range
#    (~3.0-6.0) with no inherent skewness — no transformation needed
#
# 2. Levene's test confirms unequal variances between groups
#    (BULK SD=0.09, FSW SD=0.48, 2216 SD=0.10). Welch's t-test
#    accounts for unequal variances (unlike Student's t)
#
# 3. Mann-Whitney and permutation tests face a hard combinatorial
#    ceiling for n=3 vs n=4: minimum achievable two-sided p = 0.057.
#    Welch's t-test does not have this limitation and can detect
#    the BULK vs 2216 difference (p < 0.001)
#
# 4. Cohen's d is computed to quantify effect size — especially
#    important for BULK vs FSW (p = 0.069, n.s., but d = 1.90)
#    where sample size limits statistical power
# ---------------------------------------------------------------
from scipy.stats import ttest_ind, levene, kruskal

# Extract Shannon values per group
shannon_bulk = summary_df[summary_df["Group"] == "BULK"]["Shannon_diversity"].values
shannon_fsw  = summary_df[summary_df["Group"] == "FSW"]["Shannon_diversity"].values
shannon_2216 = summary_df[summary_df["Group"] == "2216"]["Shannon_diversity"].values

def cohens_d(a, b):
    """
    Compute Cohen's d (effect size) for two independent groups.
    Pooled SD = sqrt(average of within-group variances).
    Interpretation: 0.2=small, 0.5=medium, 0.8=large
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return abs(a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

# Overall Kruskal-Wallis (kept for completeness)
kw_stat, kw_pvalue = kruskal(shannon_bulk, shannon_fsw, shannon_2216)

print("\n===== ALPHA-DIVERSITY STATISTICS =====")
print(f"\nKruskal-Wallis (all groups): H = {kw_stat:.3f}, p = {kw_pvalue:.4f}")

# Levene test for equal variances
lev_stat, lev_p = levene(shannon_bulk, shannon_fsw, shannon_2216)
print(f"Levene test (equal variances): F = {lev_stat:.3f}, p = {lev_p:.4f}")
print(f"  -> {'Variances unequal — Welch correction applied' if lev_p < 0.05 else 'Variances appear equal'}")

# Pairwise Welch's t-tests
print(f"\nPairwise Welch's t-tests:")
pairs_alpha = [
    ("BULK", "FSW",  shannon_bulk, shannon_fsw),
    ("FSW",  "2216", shannon_fsw,  shannon_2216),
    ("BULK", "2216", shannon_bulk, shannon_2216),
]

pairwise_alpha = {}
pairwise_d     = {}

for name_a, name_b, vals_a, vals_b in pairs_alpha:
    t_stat, pval = ttest_ind(vals_a, vals_b, equal_var=False)
    d            = cohens_d(vals_a, vals_b)
    d_interp     = "large" if d > 0.8 else "medium" if d > 0.5 else "small"
    pairwise_alpha[f"{name_a} vs {name_b}"] = pval
    pairwise_d[f"{name_a} vs {name_b}"]     = d
    print(f"  {name_a} vs {name_b}: t={t_stat:.3f}, p={pval:.4f}, Cohen's d={d:.2f} ({d_interp})")

# ── Plotting settings ─────────────────────────────────────────
group_order_alpha   = ["BULK", "FSW", "2216"]
group_colors_alpha  = {"BULK": "#1f77b4", "FSW": "#ff7f0e", "2216": "#2ca02c"}
group_alpha_opacity = {"BULK": 0.20, "FSW": 0.60, "2216": 0.60}
group_point_opacity = {"BULK": 0.35, "FSW": 1.0,  "2216": 1.0}
data_alpha          = [shannon_bulk, shannon_fsw, shannon_2216]

fig, ax = plt.subplots(figsize=(10, 10))

bp = ax.boxplot(
    data_alpha,
    positions=range(len(group_order_alpha)),
    widths=0.45,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=6)
)

for patch, group in zip(bp["boxes"], group_order_alpha):
    patch.set_facecolor(group_colors_alpha[group])
    patch.set_alpha(group_alpha_opacity[group])

for element in ["whiskers", "caps", "medians"]:
    for i, line in enumerate(bp[element]):
        group_idx = i // 2
        if group_order_alpha[group_idx] == "BULK":
            line.set_alpha(0.35)

np.random.seed(42)
for i, (vals, group) in enumerate(zip(data_alpha, group_order_alpha)):
    jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(
        i + jitter, vals,
        color=group_colors_alpha[group],
        alpha=group_point_opacity[group],
        s=120, zorder=5,
        edgecolors="black", linewidths=0.8
    )

def add_significance_bracket(ax, x1, x2, y, pval):
    """Draw a significance bracket. Symbol only — no Cohen's d on plot."""
    if pval < 0.001:   label = "***"
    elif pval < 0.01:  label = "**"
    elif pval < 0.05:  label = "*"
    else:              label = "n.s."
    h = 0.05
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black")
    ax.text((x1+x2)/2, y+h+0.01, label,
            ha="center", va="bottom", fontsize=24, color="black")

y_max    = max([v.max() for v in data_alpha])
y_offset = y_max * 0.08

# Show all three brackets (BULK vs FSW, FSW vs 2216, BULK vs 2216)
# positions: BULK=0, FSW=1, 2216=2
bracket_pairs = [
    (0, 1, pairwise_alpha["BULK vs FSW"],  y_max + y_offset * 0.5),
    (1, 2, pairwise_alpha["FSW vs 2216"],  y_max + y_offset * 2.0),
    (0, 2, pairwise_alpha["BULK vs 2216"], y_max + y_offset * 3.8),
]

for x1, x2, pval, y in bracket_pairs:
    add_significance_bracket(ax, x1, x2, y, pval)

ax.text(
    0, y_max + y_offset * 0.00001,
    "ambient\nreference",
    ha="center", va="top",
    fontsize=20, color=group_colors_alpha["BULK"],
    alpha=1, style="italic"
)

ax.set_xticks(range(len(group_order_alpha)))
ax.set_xticklabels(group_order_alpha, size=26)
ax.set_ylabel("Shannon diversity index", size=26)
ax.set_title("Alpha-diversity comparison between groups\n(Welch's t-test)", size=24)
plt.yticks(size=22)

ax.set_ylim(
    min([v.min() for v in data_alpha]) * 0.9,
    y_max + y_offset * 5.5
)

plt.tight_layout()
plt.savefig("alpha_diversity_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved alpha-diversity comparison plot to: alpha_diversity_comparison.png")

#%%
# -*- coding: utf-8 -*-
"""
STEP 3 of Naples dataset 16S analysis: Rarefaction curves

Goal:
- Assess whether sequencing depth was sufficient to capture bacterial diversity
  in each sample
- Plot rarefaction curves for all BULK, FSW, and 2216 samples
- Confirm that curves approach saturation, supporting downstream community comparisons

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- rarefaction_curves.png
- rarefaction_table.csv
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_FIG = Path("rarefaction_curves.png")
OUTPUT_TABLE = Path("rarefaction_table.csv")

# Number of subsampling points along the curve
N_POINTS = 20

# Number of repeated random subsamplings at each depth
N_REPEATS = 20

# Minimum reads required for a sample to be included
MIN_READS = 1000

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col = "Genus"

sample_cols = [col for col in df.columns if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. DEFINE SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

print("\nSample metadata:")
print(sample_metadata)

# =========================
# 5. RAREFACTION FUNCTION
# =========================
def rarefy_counts(counts, depth, rng):
    """
    Subsample reads without replacement down to 'depth'
    and return the number of observed ZOTUs.

    counts = vector of ZOTU counts for one sample
    depth = number of reads to subsample
    rng = numpy random generator
    """
    counts = np.array(counts, dtype=int)

    total_reads = counts.sum()
    if depth > total_reads:
        return np.nan

    # Expand counts into read-level ZOTU identities
    # Example: counts [3,2,0] -> [0,0,0,1,1]
    zotu_ids = np.repeat(np.arange(len(counts)), counts)

    # Randomly sample reads without replacement
    sampled = rng.choice(zotu_ids, size=depth, replace=False)

    # Count how many unique ZOTUs are present
    observed = len(np.unique(sampled))
    return observed

# =========================
# 6. BUILD RAREFACTION TABLE
# =========================
rng = np.random.default_rng(seed=42)

rarefaction_rows = []

for sample in sample_cols:
    counts = df[sample].to_numpy()
    total_reads = counts.sum()

    print(f"\nProcessing sample: {sample} (total reads = {total_reads})")

    if total_reads < MIN_READS:
        print(f"Skipping {sample}: fewer than {MIN_READS} reads")
        continue

    # Depths at which to evaluate rarefaction
    depths = np.linspace(100, total_reads, N_POINTS, dtype=int)
    depths = np.unique(depths)

    for depth in depths:
        observed_list = []

        for _ in range(N_REPEATS):
            observed = rarefy_counts(counts, depth, rng)
            observed_list.append(observed)

        mean_observed = np.mean(observed_list)
        sd_observed = np.std(observed_list)

        rarefaction_rows.append({
            "Sample": sample,
            "Group": sample_metadata.loc[sample_metadata["Sample"] == sample, "Group"].values[0],
            "Depth": depth,
            "Mean_observed_ZOTUs": mean_observed,
            "SD_observed_ZOTUs": sd_observed,
            "Total_reads": total_reads
        })

rarefaction_df = pd.DataFrame(rarefaction_rows)

print("\nRarefaction table:")
print(rarefaction_df.head())

rarefaction_df.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved rarefaction table to: {OUTPUT_TABLE}")

# =========================
# =========================
# 7. PLOT RAREFACTION CURVES
# =========================
# Individual sample curves are plotted in the group colour.
# The legend shows only three entries (one per group) with a
# thicker representative line — this avoids 11 legend entries
# and makes the group identity immediately clear.

group_colors = {
    "BULK": "tab:blue",
    "FSW":  "tab:orange",
    "2216": "tab:green"
}

fig, ax = plt.subplots(figsize=(20, 16))

# Plot each individual sample curve without a label
# (label=None suppresses it from the auto-legend)
for sample in rarefaction_df["Sample"].unique():
    temp  = rarefaction_df[rarefaction_df["Sample"] == sample].copy()
    group = temp["Group"].iloc[0]
    color = group_colors.get(group, "gray")

    ax.plot(
        temp["Depth"],
        temp["Mean_observed_ZOTUs"],
        label=None,          # no per-sample legend entry
        color=color,
        linewidth=1,
        alpha=0.6
    )

    ax.fill_between(
        temp["Depth"],
        temp["Mean_observed_ZOTUs"] - temp["SD_observed_ZOTUs"],
        temp["Mean_observed_ZOTUs"] + temp["SD_observed_ZOTUs"],
        color=color,
        alpha=0.4
    )

# Build three manual legend entries — one thick line per group
import matplotlib.lines as mlines

legend_handles = [
    mlines.Line2D([], [],
                  color=group_colors[g],
                  linewidth=5,
                  label=g)
    for g in ["FSW", "BULK", "2216"]
]

ax.legend(
    handles=legend_handles,
    loc="lower right",
    borderaxespad=1,
    fontsize=28
)

ax.set_xlabel("Number of reads subsampled", size=28)
ax.set_ylabel("Observed ZOTUs", size=28)
ax.set_title("Rarefaction curves for filtered bacterial communities", size=30)

plt.xscale("linear")
plt.xticks(size=28)
plt.yticks(size=28)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure to: {OUTPUT_FIG}")


#%%
# -*- coding: utf-8 -*-
"""
STEP 4 of Naples dataset 16S analysis: Family-level stacked bar plot

Goal:
- Visualize bacterial community composition at family level for each sample group
- Compare which families dominate BULK, FSW, and 2216 communities
- Provide a descriptive figure supporting the community-level differences
  detected in the beta-diversity analysis
- Family level is preferred over genus level for marine 16S data because
  many taxa are not classified to genus, giving cleaner and more consistent labels

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- family_relative_abundance_top10.csv
- stacked_barplot_family.png
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE   = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_TABLE = Path("family_relative_abundance_top10.csv")
OUTPUT_FIG   = Path("stacked_barplot_family.png")

# Top N genera to display individually (rest collapsed into "Other")
TOP_N = 15

# Group order on the x-axis
GROUPS_TO_KEEP = ["FSW", "BULK", "2216"]

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [col for col in df.columns
               if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

# Keep only the three biological groups
sample_metadata = sample_metadata[
    sample_metadata["Group"].isin(GROUPS_TO_KEEP)
].copy().reset_index(drop=True)

print("\nSamples used for stacked bar plot:")
print(sample_metadata)

# =========================
# 5. SUM COUNTS WITHIN EACH GROUP
# =========================
# Replicate samples within each group are summed together.
# This gives one pooled community profile per group.

# =========================
# EXTRACT FAMILY FROM TAXONOMY
# =========================
# We work at family level rather than genus level because:
# - Many marine taxa are not classified down to genus
# - Family level gives cleaner, more consistent labels
# - It is more commonly used in published marine microbiome figures
#
# If family is missing, we fall back to order level.
# This avoids losing reads that are classified only to order or above.

def extract_family(tax_string):
    """
    Extract family from the SILVA taxonomy string.
    Falls back to order level if family is not available.
    """
    if pd.isna(tax_string):
        return "Unclassified"

    parts = [p.strip() for p in str(tax_string).split(";") if p.strip()]

    # Try family first
    for part in parts:
        if part.startswith("f__"):
            family = part.replace("f__", "").strip()
            if family and family.lower() not in ["uncultured", "unclassified", "unknown", "metagenome"]:
                return family

    # Fallback to order
    for part in parts:
        if part.startswith("o__"):
            order = part.replace("o__", "").strip()
            if order and order.lower() not in ["uncultured", "unclassified", "unknown", "metagenome"]:
                return f"{order} (order)"

    return "Unclassified"

df["Family"] = df[taxonomy_col].apply(extract_family)

print("\nExample taxonomy-to-family assignments:")
print(df[[taxonomy_col, "Family"]].head(15))

# Column name used for grouping
tax_col = "Family"

grouped_counts = pd.DataFrame({
    tax_col: df[tax_col]
})

for group in GROUPS_TO_KEEP:
    group_samples = sample_metadata.loc[
        sample_metadata["Group"] == group, "Sample"
    ].tolist()

    grouped_counts[group] = df[group_samples].sum(axis=1)

print("\nGrouped counts table (first rows):")
print(grouped_counts.head())

# =========================
# 6. SUM COUNTS BY FAMILY
# =========================
# Multiple ZOTUs can share the same family label.
# Here we merge all ZOTUs with the same family into one row.

genus_counts = grouped_counts.groupby(tax_col, as_index=False)[GROUPS_TO_KEEP].sum()

print(f"\nNumber of unique families: {len(genus_counts)}")

# =========================
# 7. CONVERT TO RELATIVE ABUNDANCE
# =========================
# Each group is divided by its total so values sum to 1.
# This allows fair comparison between groups regardless of read depth.

genus_relative = genus_counts.copy()

for group in GROUPS_TO_KEEP:
    total = genus_relative[group].sum()
    genus_relative[group] = genus_relative[group] / total

# =========================
# 8. SELECT TOP 10 GENERA
# =========================
# Rank genera by their mean relative abundance across all three groups.
# Keep the top 10; collapse the rest into "Other".

genus_relative["Mean_abundance"] = genus_relative[GROUPS_TO_KEEP].mean(axis=1)

top_genera = (
    genus_relative
    .sort_values("Mean_abundance", ascending=False)
    .head(TOP_N)
    .copy()
)

other_genera = genus_relative[
    ~genus_relative[tax_col].isin(top_genera[tax_col])
].copy()

# Build "Other" row
other_row = pd.DataFrame({
    tax_col: ["Other"],
    **{group: [other_genera[group].sum()] for group in GROUPS_TO_KEEP}
})

# Combine top families + Other
plot_df = pd.concat(
    [top_genera[[tax_col] + GROUPS_TO_KEEP], other_row],
    ignore_index=True
)

# Sort by total abundance so most abundant family is at the bottom of each bar
plot_df["Total"] = plot_df[GROUPS_TO_KEEP].sum(axis=1)
plot_df = plot_df.sort_values("Total", ascending=True).drop(columns="Total")
plot_df = plot_df.reset_index(drop=True)

# Save table
plot_df.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved family relative abundance table to: {OUTPUT_TABLE}")

print("\nTable used for plotting:")
print(plot_df)

# =========================
# 9. BUILD COLOR PALETTE
# =========================
# Use a colormap to assign one distinct color per taxon.
# "Other" is always gray.

import matplotlib.cm as cm
import numpy as np

n_taxa  = len(plot_df)
cmap    = cm.get_cmap("tab20", n_taxa)
colors  = [cmap(i) for i in range(n_taxa)]

# Force "Other" to be gray regardless of position
taxa_list = plot_df[tax_col].tolist()
color_map = {}
gray_idx  = taxa_list.index("Other")

for i, taxon in enumerate(taxa_list):
    if taxon == "Other":
        color_map[taxon] = "#BEBEBE"
    else:
        color_map[taxon] = colors[i]

# =========================
# 10. PLOT STACKED BAR PLOT
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

# Reformat: rows = groups, columns = families
plot_long = plot_df.set_index(tax_col)[GROUPS_TO_KEEP].T
plot_long = plot_long.loc[GROUPS_TO_KEEP]

# Plot each family as one stacked layer
bottom = np.zeros(len(GROUPS_TO_KEEP))

for taxon in plot_df[tax_col].tolist():
    values = plot_long[taxon].values
    ax.bar(
        GROUPS_TO_KEEP,
        values,
        bottom=bottom,
        color=color_map[taxon],
        label=taxon,
        width=0.55,
        edgecolor="white",
        linewidth=0.4
    )
    bottom += values

ax.set_ylabel("Relative abundance", size=22)
ax.set_xlabel("Sample group", size=22)
ax.set_title(
    "Family-level relative abundance\n(Bacteria + Archaea, contaminant genera removed)",
    size=22
)
plt.xticks(size=20)
plt.yticks(size=20)
ax.set_ylim(0, 1)

# Legend — reversed so order matches bar stacking (most abundant at bottom)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    title="Family",
    title_fontsize=14,
    fontsize=13,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved stacked bar plot to: {OUTPUT_FIG}")

#%%
# -*- coding: utf-8 -*-
"""
STEP 5 of Naples dataset 16S analysis: Enrichment analysis — 2216 vs FSW

Goal:
- Identify which bacterial families are specifically enriched in 2216
  chemoattractant wells compared to FSW passive-entry control wells
- These enriched families represent putative chemotaxis-selected taxa
- Analysis is performed at family level, consistent with Step 4

Approach:
- Compute relative abundance per replicate for each family
- Calculate mean relative abundance in 2216 and FSW
- Calculate log2 fold-change (2216 / FSW) as effect size
- Test for significant differences using Mann-Whitney U test
  (non-parametric, appropriate for small n and non-normal distributions)

IMPORTANT CAVEAT on statistical power:
  With n=4 replicates per group, the Mann-Whitney test has very limited
  power. The minimum achievable two-sided p-value for n=4 vs n=4 is
  approximately 0.029. This means that even perfectly separated groups
  may not always reach p < 0.05. For this reason, both effect size
  (log2 fold-change) and consistency across replicates should be
  interpreted alongside the p-value, not replaced by it.

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- enrichment_2216_vs_FSW.csv      (full results table)
- enrichment_plot_2216_vs_FSW.png (horizontal bar plot)
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE    = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_TABLE  = Path("enrichment_2216_vs_FSW.csv")
OUTPUT_FIG    = Path("enrichment_plot_2216_vs_FSW.png")

# Significance threshold
ALPHA = 0.05

# Pseudocount added before log2 transformation to avoid log(0)
# Using a small value relative to typical relative abundances
PSEUDOCOUNT = 1e-6

# Minimum mean relative abundance in at least one group
# to be included in the analysis (filters out ultra-rare families)
MIN_MEAN_ABUNDANCE = 0.001   # = 0.1%

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [col for col in df.columns
               if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

# Keep only 2216 and FSW for this comparison
sample_metadata = sample_metadata[
    sample_metadata["Group"].isin(["2216", "FSW"])
].copy().reset_index(drop=True)

samples_2216 = sample_metadata[sample_metadata["Group"] == "2216"]["Sample"].tolist()
samples_fsw  = sample_metadata[sample_metadata["Group"] == "FSW"]["Sample"].tolist()

print(f"\n2216 replicates ({len(samples_2216)}): {samples_2216}")
print(f"FSW  replicates ({len(samples_fsw)}):  {samples_fsw}")

# =========================
# 5. EXTRACT FAMILY FROM TAXONOMY
# =========================
# Same function as in Step 4 — extracts family level,
# falls back to order if family is missing.

def extract_family(tax_string):
    if pd.isna(tax_string):
        return "Unclassified"
    parts = [p.strip() for p in str(tax_string).split(";") if p.strip()]
    for part in parts:
        if part.startswith("f__"):
            family = part.replace("f__", "").strip()
            if family and family.lower() not in ["uncultured", "unclassified", "unknown", "metagenome"]:
                return family
    for part in parts:
        if part.startswith("o__"):
            order = part.replace("o__", "").strip()
            if order and order.lower() not in ["uncultured", "unclassified", "unknown", "metagenome"]:
                return f"{order} (order)"
    return "Unclassified"

df["Family"] = df[taxonomy_col].apply(extract_family)

# =========================
# 6. COMPUTE PER-REPLICATE RELATIVE ABUNDANCE AT FAMILY LEVEL
# =========================
# For each sample, we compute relative abundance per ZOTU first,
# then sum by family. This gives one value per family per replicate.
#
# This is important: we do NOT pool replicates here.
# We keep each replicate separate so the Mann-Whitney test
# can compare the distributions across the 4 replicates per group.

all_samples = samples_2216 + samples_fsw

# Compute relative abundance per sample
rel_abundance = df[all_samples].copy()
rel_abundance = rel_abundance.div(rel_abundance.sum(axis=0), axis=1)

# Add family column
rel_abundance["Family"] = df["Family"].values

# Sum by family within each sample
family_rel = rel_abundance.groupby("Family")[all_samples].sum()

print(f"\nFamily-level relative abundance matrix shape: {family_rel.shape}")
print("(rows = families, columns = samples)")

# =========================
# 7. FILTER LOW-ABUNDANCE FAMILIES
# =========================
# Remove families that are extremely rare in both groups,
# as these are not biologically meaningful for enrichment analysis.

mean_2216 = family_rel[samples_2216].mean(axis=1)
mean_fsw  = family_rel[samples_fsw].mean(axis=1)

keep = (mean_2216 >= MIN_MEAN_ABUNDANCE) | (mean_fsw >= MIN_MEAN_ABUNDANCE)
family_rel = family_rel[keep].copy()

print(f"\nFamilies after abundance filter (>= {MIN_MEAN_ABUNDANCE}): {len(family_rel)}")

# =========================
# 8. COMPUTE FOLD-CHANGE AND STATISTICAL TEST
# =========================
# For each family:
# - mean_2216 = mean relative abundance across 2216 replicates
# - mean_fsw  = mean relative abundance across FSW replicates
# - log2FC    = log2((mean_2216 + pseudocount) / (mean_fsw + pseudocount))
#               positive = enriched in 2216
#               negative = depleted in 2216 (enriched in FSW)
# - p-value   = Mann-Whitney U test (two-sided)

results = []

for family in family_rel.index:
    vals_2216 = family_rel.loc[family, samples_2216].values.astype(float)
    vals_fsw  = family_rel.loc[family, samples_fsw].values.astype(float)

    mean_2216_val = np.mean(vals_2216)
    mean_fsw_val  = np.mean(vals_fsw)

    # Log2 fold-change with pseudocount to avoid log(0)
    log2fc = np.log2(
        (mean_2216_val + PSEUDOCOUNT) / (mean_fsw_val + PSEUDOCOUNT)
    )

    # Mann-Whitney U test (two-sided, non-parametric)
    # With n=4 vs n=4 the minimum achievable p-value is ~0.029
    try:
        stat, pval = mannwhitneyu(vals_2216, vals_fsw, alternative="two-sided")
    except ValueError:
        # Can happen if all values are identical
        pval = 1.0

    results.append({
        "Family":          family,
        "Mean_2216":       round(mean_2216_val, 6),
        "Mean_FSW":        round(mean_fsw_val,  6),
        "Log2FC":          round(log2fc, 3),
        "p_value":         round(pval, 4),
        "Significant":     pval < ALPHA,
        "Direction":       "Enriched in 2216" if log2fc > 0 else "Enriched in FSW"
    })

results_df = pd.DataFrame(results)

# Sort by mean abundance in 2216, then by log2FC
results_df = results_df.sort_values(
    ["Mean_2216", "Log2FC"],
    ascending=[False, False]
).reset_index(drop=True)

# Save full table
results_df.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved enrichment table to: {OUTPUT_TABLE}")

# Print summary
n_enriched  = (results_df["Log2FC"] > 0).sum()
n_depleted  = (results_df["Log2FC"] < 0).sum()
n_sig       = results_df["Significant"].sum()

print(f"\nTotal families tested : {len(results_df)}")
print(f"Enriched in 2216     : {n_enriched}")
print(f"Enriched in FSW      : {n_depleted}")
print(f"Significant (p<{ALPHA}) : {n_sig}")
print("\nTop 10 families enriched in 2216:")
print(
    results_df[results_df["Log2FC"] > 0]
    .head(10)
    [["Family", "Mean_2216", "Mean_FSW", "Log2FC", "p_value", "Significant"]]
    .to_string(index=False)
)

# =========================
# 9. PLOT — HORIZONTAL BAR PLOT
# =========================
# Shows log2 fold-change for top enriched families in 2216
# Bars are sorted by mean abundance in 2216 (most abundant at top)
# Color: green = enriched in 2216, orange = enriched in FSW
# Significance: * on bars that pass p < 0.05

# Select top families to plot:
# top 10 enriched in 2216 + top 5 enriched in FSW
top_2216 = results_df[results_df["Log2FC"] > 0].head(10)
top_fsw  = results_df[results_df["Log2FC"] < 0].sort_values("Log2FC").head(5)

plot_data = pd.concat([top_2216, top_fsw], ignore_index=True)

# Sort by log2FC for the horizontal bar plot
plot_data = plot_data.sort_values("Log2FC", ascending=True).reset_index(drop=True)

# Colors
bar_colors = [
    "#2ca02c" if fc > 0 else "#ff7f0e"
    for fc in plot_data["Log2FC"]
]

fig, ax = plt.subplots(figsize=(10, 8))

bars = ax.barh(
    plot_data["Family"],
    plot_data["Log2FC"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
    height=0.6
)

# Add significance marker (*) on bars that pass threshold
for i, (_, row) in enumerate(plot_data.iterrows()):
    if row["Significant"]:
        x_pos = row["Log2FC"] + (0.1 if row["Log2FC"] > 0 else -0.1)
        ha    = "left" if row["Log2FC"] > 0 else "right"
        ax.text(x_pos, i, "*", va="center", ha=ha, fontsize=16, color="black")

# Reference line at 0
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("Log2 fold-change (2216 / FSW)", size=16)
ax.set_title(
    "Family-level enrichment: 2216 vs FSW\n"
    "* p < 0.05 (Mann-Whitney U test, two-sided)",
    size=16
)
plt.xticks(size=13)
plt.yticks(size=13)

# Legend
legend_patches = [
    mpatches.Patch(color="#2ca02c", label="Enriched in 2216"),
    mpatches.Patch(color="#ff7f0e", label="Enriched in FSW")
]
ax.legend(handles=legend_patches, fontsize=13, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved enrichment plot to: {OUTPUT_FIG}")

# =========================
# 10. INTERPRETATION GUIDE
# =========================
print("\n===== INTERPRETATION GUIDE =====")
print()
print("Log2FC > 0  → family is more abundant in 2216 than FSW")
print("Log2FC < 0  → family is more abundant in FSW than 2216")
print("Log2FC = 1  → 2x more abundant in 2216")
print("Log2FC = 2  → 4x more abundant in 2216")
print("Log2FC = -1 → 2x more abundant in FSW")
print()
print("* = significant at p < 0.05 (Mann-Whitney U test)")
print()
print("CAVEAT: with n=4 vs n=4, the minimum achievable p-value")
print("is ~0.029. Interpret effect size (Log2FC) and consistency")
print("across replicates alongside the p-value.")

#%%
# -*- coding: utf-8 -*-
"""
STEP 6 of Naples dataset 16S analysis: Genus-level enrichment analysis — 2216 vs FSW

Goal:
- Zoom into the family-level enrichment results (Step 5) at genus resolution
- Identify which specific genera within enriched families are driving the signal
- Being able to name specific genera strengthens the biological interpretation
  considerably compared to family level alone
- For example: knowing Nitrospinaceae is enriched (Step 5) is good;
  knowing it is specifically Nitrospina that is enriched is better

Approach:
- Same statistical framework as Step 5 (Mann-Whitney U, log2 fold-change)
- Genus level: uses g__ prefix from SILVA taxonomy
- Fallback to family level if genus is missing

IMPORTANT CAVEAT on statistical power:
  With n=4 replicates per group, the minimum achievable two-sided
  p-value is approximately 0.029. Effect size (log2FC) and consistency
  across replicates should be interpreted alongside the p-value.

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- enrichment_genus_2216_vs_FSW.csv       (full results table)
- enrichment_plot_genus_2216_vs_FSW.png  (horizontal bar plot)
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import mannwhitneyu

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE    = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_TABLE  = Path("enrichment_genus_2216_vs_FSW.csv")
OUTPUT_FIG    = Path("enrichment_plot_genus_2216_vs_FSW.png")

ALPHA              = 0.05
PSEUDOCOUNT        = 1e-6
MIN_MEAN_ABUNDANCE = 0.001   # 0.1% minimum mean abundance in at least one group

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [col for col in df.columns
               if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)
sample_metadata = sample_metadata[
    sample_metadata["Group"].isin(["2216", "FSW"])
].copy().reset_index(drop=True)

samples_2216 = sample_metadata[sample_metadata["Group"] == "2216"]["Sample"].tolist()
samples_fsw  = sample_metadata[sample_metadata["Group"] == "FSW"]["Sample"].tolist()

print(f"\n2216 replicates ({len(samples_2216)}): {samples_2216}")
print(f"FSW  replicates ({len(samples_fsw)}):  {samples_fsw}")

# =========================
# 5. EXTRACT GENUS FROM TAXONOMY
# =========================
# We use the Genus column already computed in Step 1.
# That column uses g__ prefix with fallback to f__ (family) or o__ (order).
# This is already available in the filtered table.
#
# The "Genus" column in the filtered file already contains cleaned labels,
# so we can use it directly without re-extracting from taxonomy.

print("\nExample genus labels in filtered table:")
print(df[genus_col].value_counts().head(10))

# =========================
# 6. COMPUTE PER-REPLICATE RELATIVE ABUNDANCE AT GENUS LEVEL
# =========================
# Each replicate is kept separate for the Mann-Whitney test.
# We do NOT pool replicates here.

all_samples = samples_2216 + samples_fsw

rel_abundance = df[all_samples].copy()
rel_abundance = rel_abundance.div(rel_abundance.sum(axis=0), axis=1)
rel_abundance[genus_col] = df[genus_col].values

genus_rel = rel_abundance.groupby(genus_col)[all_samples].sum()

print(f"\nGenus-level relative abundance matrix: {genus_rel.shape}")
print("(rows = genera, columns = samples)")

# =========================
# 7. FILTER LOW-ABUNDANCE GENERA
# =========================
mean_2216 = genus_rel[samples_2216].mean(axis=1)
mean_fsw  = genus_rel[samples_fsw].mean(axis=1)

keep      = (mean_2216 >= MIN_MEAN_ABUNDANCE) | (mean_fsw >= MIN_MEAN_ABUNDANCE)
genus_rel = genus_rel[keep].copy()

print(f"\nGenera after abundance filter (>= {MIN_MEAN_ABUNDANCE}): {len(genus_rel)}")

# =========================
# 8. COMPUTE FOLD-CHANGE AND MANN-WHITNEY TEST
# =========================
results = []

for genus in genus_rel.index:
    vals_2216 = genus_rel.loc[genus, samples_2216].values.astype(float)
    vals_fsw  = genus_rel.loc[genus, samples_fsw].values.astype(float)

    mean_2216_val = np.mean(vals_2216)
    mean_fsw_val  = np.mean(vals_fsw)

    log2fc = np.log2(
        (mean_2216_val + PSEUDOCOUNT) / (mean_fsw_val + PSEUDOCOUNT)
    )

    try:
        _, pval = mannwhitneyu(vals_2216, vals_fsw, alternative="two-sided")
    except ValueError:
        pval = 1.0

    results.append({
        "Genus":       genus,
        "Mean_2216":   round(mean_2216_val, 6),
        "Mean_FSW":    round(mean_fsw_val,  6),
        "Log2FC":      round(log2fc, 3),
        "p_value":     round(pval, 4),
        "Significant": pval < ALPHA,
        "Direction":   "Enriched in 2216" if log2fc > 0 else "Enriched in FSW"
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["Mean_2216", "Log2FC"],
    ascending=[False, False]
).reset_index(drop=True)

results_df.to_csv(OUTPUT_TABLE, index=False)
print(f"\nSaved genus enrichment table to: {OUTPUT_TABLE}")

# Summary
n_enriched = (results_df["Log2FC"] > 0).sum()
n_depleted = (results_df["Log2FC"] < 0).sum()
n_sig      = results_df["Significant"].sum()

print(f"\nTotal genera tested  : {len(results_df)}")
print(f"Enriched in 2216     : {n_enriched}")
print(f"Enriched in FSW      : {n_depleted}")
print(f"Significant (p<{ALPHA}) : {n_sig}")
print("\nTop 15 genera enriched in 2216:")
print(
    results_df[results_df["Log2FC"] > 0]
    .head(15)
    [["Genus", "Mean_2216", "Mean_FSW", "Log2FC", "p_value", "Significant"]]
    .to_string(index=False)
)

# =========================
# 9. PLOT — HORIZONTAL BAR PLOT
# =========================
# Top 15 enriched in 2216 + top 5 enriched in FSW
# Sorted by log2FC for readability
# * marks genera that pass p < 0.05

top_2216 = results_df[results_df["Log2FC"] > 0].head(15)
top_fsw  = results_df[results_df["Log2FC"] < 0].sort_values("Log2FC").head(5)

plot_data = pd.concat([top_2216, top_fsw], ignore_index=True)
plot_data = plot_data.sort_values("Log2FC", ascending=True).reset_index(drop=True)

bar_colors = [
    "#2ca02c" if fc > 0 else "#ff7f0e"
    for fc in plot_data["Log2FC"]
]

fig, ax = plt.subplots(figsize=(11, 10))

ax.barh(
    plot_data["Genus"],
    plot_data["Log2FC"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
    height=0.6
)

# Significance markers
for i, (_, row) in enumerate(plot_data.iterrows()):
    if row["Significant"]:
        x_pos = row["Log2FC"] + (0.05 if row["Log2FC"] > 0 else -0.05)
        ha    = "left" if row["Log2FC"] > 0 else "right"
        ax.text(x_pos, i, "*", va="center", ha=ha, fontsize=16, color="black")

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Log2 fold-change (2216 / FSW)", size=16)
ax.set_title(
    "Genus-level enrichment: 2216 vs FSW\n"
    "* p < 0.05 (Mann-Whitney U test, two-sided)",
    size=16
)
plt.xticks(size=13)
plt.yticks(size=12)

legend_patches = [
    mpatches.Patch(color="#2ca02c", label="Enriched in 2216"),
    mpatches.Patch(color="#ff7f0e", label="Enriched in FSW")
]
ax.legend(handles=legend_patches, fontsize=13, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved genus enrichment plot to: {OUTPUT_FIG}")

# =========================
# 10. INTERPRETATION GUIDE
# =========================
print("\n===== INTERPRETATION GUIDE =====")
print()
print("Log2FC > 0  → genus is more abundant in 2216 than FSW")
print("Log2FC < 0  → genus is more abundant in FSW than 2216")
print("Log2FC = 1  → 2x more abundant in 2216")
print("Log2FC = 2  → 4x more abundant in 2216")
print()
print("Compare these genus-level results with the family-level")
print("enrichment (Step 5) to identify which genera within")
print("enriched families are driving the signal.")
print()
print("* = significant at p < 0.05 (Mann-Whitney U test)")
print()
print("CAVEAT: with n=4 vs n=4, the minimum achievable p-value")
print("is ~0.029. Interpret effect size and consistency across")
print("replicates alongside the p-value.")

#%%
# -*- coding: utf-8 -*-
"""
STEP 7 of Naples dataset 16S analysis: Phylum-level taxa abundance and enrichment analysis FSW vs 2216

Goal:
- Provide a broad overview of bacterial community composition at phylum level
- Visualize relative abundance of top 10 phyla across BULK, FSW, and 2216
- Test which phyla are significantly enriched in 2216 vs FSW
- Phylum level is useful as a broad contextual figure (e.g. supplementary)
  before the finer family-level analysis

Note:
- Phylum level gives cleaner labels than family/genus
- The main chemotaxis story lives at family level (Step 4/5)
- This step provides the broad ecological context

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- phylum_relative_abundance_top10.csv
- stacked_barplot_phylum.png
- enrichment_phylum_2216_vs_FSW.csv
- enrichment_plot_phylum_2216_vs_FSW.png
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import mannwhitneyu

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE        = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_TABLE_BAR  = Path("phylum_relative_abundance_top10.csv")
OUTPUT_FIG_BAR    = Path("stacked_barplot_phylum.png")
OUTPUT_TABLE_ENR  = Path("enrichment_phylum_2216_vs_FSW.csv")
OUTPUT_FIG_ENR    = Path("enrichment_plot_phylum_2216_vs_FSW.png")

TOP_N             = 15
GROUPS_BAR        = ["FSW", "BULK", "2216"]   # x-axis order for bar plot
ALPHA             = 0.05
PSEUDOCOUNT       = 1e-6
MIN_MEAN_ABUNDANCE = 0.001

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [col for col in df.columns
               if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

sample_metadata_bar = sample_metadata[
    sample_metadata["Group"].isin(GROUPS_BAR)
].copy().reset_index(drop=True)

samples_2216 = sample_metadata[sample_metadata["Group"] == "2216"]["Sample"].tolist()
samples_fsw  = sample_metadata[sample_metadata["Group"] == "FSW"]["Sample"].tolist()

print("\nSamples for bar plot:")
print(sample_metadata_bar)

# =========================
# 5. EXTRACT PHYLUM FROM TAXONOMY
# =========================
# Phylum is the second level in SILVA taxonomy (p__).
# At this level every bacterial sequence should have a valid assignment,
# so fallbacks are rarely needed.

def extract_phylum(tax_string):
    """
    Extract phylum from the SILVA taxonomy string.
    Falls back to class if phylum is missing (very rare).
    """
    if pd.isna(tax_string):
        return "Unclassified"

    parts = [p.strip() for p in str(tax_string).split(";") if p.strip()]

    # Try phylum
    for part in parts:
        if part.startswith("p__"):
            phylum = part.replace("p__", "").strip()
            if phylum and phylum.lower() not in ["uncultured", "unclassified", "unknown"]:
                return phylum

    # Fallback to class
    for part in parts:
        if part.startswith("c__"):
            cls = part.replace("c__", "").strip()
            if cls:
                return f"{cls} (class)"

    return "Unclassified"

df["Phylum"] = df[taxonomy_col].apply(extract_phylum)

print("\nExample taxonomy-to-phylum assignments:")
print(df[[taxonomy_col, "Phylum"]].head(8))

# =========================
# 6. SUM COUNTS BY GROUP AND PHYLUM (for bar plot)
# =========================
grouped_counts = pd.DataFrame({"Phylum": df["Phylum"]})

for group in GROUPS_BAR:
    group_samples = sample_metadata_bar.loc[
        sample_metadata_bar["Group"] == group, "Sample"
    ].tolist()
    grouped_counts[group] = df[group_samples].sum(axis=1)

phylum_counts = grouped_counts.groupby("Phylum", as_index=False)[GROUPS_BAR].sum()

print(f"\nNumber of unique phyla: {len(phylum_counts)}")

# =========================
# 7. CONVERT TO RELATIVE ABUNDANCE
# =========================
phylum_relative = phylum_counts.copy()

for group in GROUPS_BAR:
    total = phylum_relative[group].sum()
    phylum_relative[group] = phylum_relative[group] / total

# =========================
# 8. SELECT TOP 10 PHYLA
# =========================
phylum_relative["Mean_abundance"] = phylum_relative[GROUPS_BAR].mean(axis=1)

top_phyla = (
    phylum_relative
    .sort_values("Mean_abundance", ascending=False)
    .head(TOP_N)
    .copy()
)

other_phyla = phylum_relative[
    ~phylum_relative["Phylum"].isin(top_phyla["Phylum"])
].copy()

other_row = pd.DataFrame({
    "Phylum": ["Other"],
    **{group: [other_phyla[group].sum()] for group in GROUPS_BAR}
})

plot_df = pd.concat(
    [top_phyla[["Phylum"] + GROUPS_BAR], other_row],
    ignore_index=True
)

plot_df["Total"] = plot_df[GROUPS_BAR].sum(axis=1)
plot_df = plot_df.sort_values("Total", ascending=True).drop(columns="Total")
plot_df = plot_df.reset_index(drop=True)

plot_df.to_csv(OUTPUT_TABLE_BAR, index=False)
print(f"\nSaved phylum abundance table to: {OUTPUT_TABLE_BAR}")
print("\nTable used for plotting:")
print(plot_df)

# =========================
# 9. BUILD COLOR PALETTE FOR BAR PLOT
# =========================
n_taxa   = len(plot_df)
cmap     = cm.get_cmap("tab20", n_taxa)
colors   = [cmap(i) for i in range(n_taxa)]
taxa_list = plot_df["Phylum"].tolist()

color_map = {}
for i, taxon in enumerate(taxa_list):
    color_map[taxon] = "#BEBEBE" if taxon == "Other" else colors[i]

# =========================
# 10. PLOT STACKED BAR PLOT
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

plot_long = plot_df.set_index("Phylum")[GROUPS_BAR].T
plot_long = plot_long.loc[GROUPS_BAR]

bottom = np.zeros(len(GROUPS_BAR))

for taxon in plot_df["Phylum"].tolist():
    values = plot_long[taxon].values
    ax.bar(
        GROUPS_BAR,
        values,
        bottom=bottom,
        color=color_map[taxon],
        label=taxon,
        width=0.55,
        edgecolor="white",
        linewidth=0.4
    )
    bottom += values

ax.set_ylabel("Relative abundance", size=22)
ax.set_xlabel("Sample group", size=22)
ax.set_title(
    "Phylum-level relative abundance\n(Bacteria + Archaea, contaminant genera removed)",
    size=22
)
plt.xticks(size=20)
plt.yticks(size=20)
ax.set_ylim(0, 1)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    title="Phylum",
    title_fontsize=14,
    fontsize=13,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.savefig(OUTPUT_FIG_BAR, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved phylum stacked bar plot to: {OUTPUT_FIG_BAR}")

# =========================
# 11. PER-REPLICATE RELATIVE ABUNDANCE (for enrichment)
# =========================
all_samples  = samples_2216 + samples_fsw

rel_abundance = df[all_samples].copy()
rel_abundance = rel_abundance.div(rel_abundance.sum(axis=0), axis=1)
rel_abundance["Phylum"] = df["Phylum"].values

phylum_rel = rel_abundance.groupby("Phylum")[all_samples].sum()

# Filter low-abundance phyla
mean_2216 = phylum_rel[samples_2216].mean(axis=1)
mean_fsw  = phylum_rel[samples_fsw].mean(axis=1)
keep      = (mean_2216 >= MIN_MEAN_ABUNDANCE) | (mean_fsw >= MIN_MEAN_ABUNDANCE)
phylum_rel = phylum_rel[keep].copy()

print(f"\nPhyla after abundance filter: {len(phylum_rel)}")

# =========================
# 12. COMPUTE FOLD-CHANGE AND MANN-WHITNEY TEST
# =========================
results = []

for phylum in phylum_rel.index:
    vals_2216 = phylum_rel.loc[phylum, samples_2216].values.astype(float)
    vals_fsw  = phylum_rel.loc[phylum, samples_fsw].values.astype(float)

    mean_2216_val = np.mean(vals_2216)
    mean_fsw_val  = np.mean(vals_fsw)

    log2fc = np.log2(
        (mean_2216_val + PSEUDOCOUNT) / (mean_fsw_val + PSEUDOCOUNT)
    )

    try:
        _, pval = mannwhitneyu(vals_2216, vals_fsw, alternative="two-sided")
    except ValueError:
        pval = 1.0

    results.append({
        "Phylum":      phylum,
        "Mean_2216":   round(mean_2216_val, 6),
        "Mean_FSW":    round(mean_fsw_val,  6),
        "Log2FC":      round(log2fc, 3),
        "p_value":     round(pval, 4),
        "Significant": pval < ALPHA,
        "Direction":   "Enriched in 2216" if log2fc > 0 else "Enriched in FSW"
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    ["Mean_2216", "Log2FC"], ascending=[False, False]
).reset_index(drop=True)

results_df.to_csv(OUTPUT_TABLE_ENR, index=False)
print(f"\nSaved phylum enrichment table to: {OUTPUT_TABLE_ENR}")

# Print summary
print(f"\nTotal phyla tested   : {len(results_df)}")
print(f"Enriched in 2216     : {(results_df['Log2FC'] > 0).sum()}")
print(f"Enriched in FSW      : {(results_df['Log2FC'] < 0).sum()}")
print(f"Significant (p<{ALPHA}) : {results_df['Significant'].sum()}")
print("\nAll phyla sorted by abundance in 2216:")
print(
    results_df[["Phylum", "Mean_2216", "Mean_FSW", "Log2FC", "p_value", "Significant"]]
    .to_string(index=False)
)

# =========================
# 13. PLOT ENRICHMENT — HORIZONTAL BAR PLOT
# =========================
top_2216 = results_df[results_df["Log2FC"] > 0].head(15)
top_fsw  = results_df[results_df["Log2FC"] < 0].sort_values("Log2FC").head(5)

plot_data = pd.concat([top_2216, top_fsw], ignore_index=True)
plot_data = plot_data.sort_values("Log2FC", ascending=True).reset_index(drop=True)

bar_colors = [
    "#2ca02c" if fc > 0 else "#ff7f0e"
    for fc in plot_data["Log2FC"]
]

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(
    plot_data["Phylum"],
    plot_data["Log2FC"],
    color=bar_colors,
    edgecolor="black",
    linewidth=0.5,
    height=0.6
)

for i, (_, row) in enumerate(plot_data.iterrows()):
    if row["Significant"]:
        x_pos = row["Log2FC"] + (0.05 if row["Log2FC"] > 0 else -0.05)
        ha    = "left" if row["Log2FC"] > 0 else "right"
        ax.text(x_pos, i, "*", va="center", ha=ha, fontsize=16, color="black")

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Log2 fold-change (2216 / FSW)", size=16)
ax.set_title(
    "Phylum-level enrichment: 2216 vs FSW\n"
    "* p < 0.05 (Mann-Whitney U test, two-sided)",
    size=16
)
plt.xticks(size=13)
plt.yticks(size=13)

legend_patches = [
    mpatches.Patch(color="#2ca02c", label="Enriched in 2216"),
    mpatches.Patch(color="#ff7f0e", label="Enriched in FSW")
]
ax.legend(handles=legend_patches, fontsize=13, loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_FIG_ENR, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved phylum enrichment plot to: {OUTPUT_FIG_ENR}")

#%%
# -*- coding: utf-8 -*-
"""
STEP 8 of Naples dataset 16S analysis: Beta-diversity calculation and plotting as PCoA analysis

Goal:
- Compare bacterial community composition between BULK, FSW, and 2216
- Use Bray-Curtis dissimilarity as the distance metric
- Visualize with PCoA (Principal Coordinates Analysis)
- Test for significant group differences with PERMANOVA
- Display overall PERMANOVA result directly on the PCoA plot

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- braycurtis_distance_matrix.csv
- pcoa_coordinates.csv
- pcoa_plot.png
- permanova_results.txt

"""
# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import squareform, pdist
from skbio.stats.ordination import pcoa
from skbio.stats.distance import DistanceMatrix, permanova

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE   = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_DIST  = Path("braycurtis_distance_matrix.csv")
OUTPUT_PCOA  = Path("pcoa_coordinates.csv")
OUTPUT_FIG   = Path("pcoa_plot.png")
OUTPUT_STATS = Path("permanova_results.txt")

GROUPS_TO_KEEP = ["BULK", "FSW", "2216"]

# Colors and marker shapes per group
GROUP_COLORS  = {"BULK": "tab:blue", "FSW": "tab:orange", "2216": "tab:green"}
GROUP_MARKERS = {"BULK": "o",        "FSW": "s",          "2216": "^"}

# PERMANOVA settings
N_PERMUTATIONS = 9999

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [col for col in df.columns
               if col not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    """
    Assign each sample to a biological group based on its name.
    """
    if sample_name.startswith("2216-"):
        return "2216"
    elif sample_name.startswith("FSW-C-"):
        return "FSW-C"
    elif sample_name.startswith("FSW-"):
        return "FSW"
    elif sample_name.startswith("BULK-"):
        return "BULK"
    elif sample_name.startswith("CTR-E-"):
        return "CTR-E"
    elif sample_name.startswith("CTR-PCR"):
        return "CTR-PCR"
    else:
        return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)

# Keep only the three biological groups
sample_metadata = sample_metadata[
    sample_metadata["Group"].isin(GROUPS_TO_KEEP)
].copy().reset_index(drop=True)

print("\nSamples kept for beta-diversity:")
print(sample_metadata)

samples_to_use = sample_metadata["Sample"].tolist()

# =========================
# 5. BUILD THE RELATIVE ABUNDANCE MATRIX
# =========================
# rows = samples, columns = ZOTUs
count_matrix = df[samples_to_use].copy()
count_matrix = count_matrix.T
count_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

print("\nRelative abundance matrix shape (samples x ZOTUs):")
print(count_matrix.shape)

# =========================
# 6. COMPUTE BRAY-CURTIS DISTANCE MATRIX
# =========================
# Bray-Curtis dissimilarity measures how different two communities are.
# Value of 0 = identical communities
# Value of 1 = completely different communities

bc_distances = pdist(count_matrix.values, metric="braycurtis")
bc_matrix = pd.DataFrame(
    squareform(bc_distances),
    index=samples_to_use,
    columns=samples_to_use
)

print("\nBray-Curtis distance matrix:")
print(bc_matrix.round(3))

bc_matrix.to_csv(OUTPUT_DIST)
print(f"\nSaved distance matrix to: {OUTPUT_DIST}")

# =========================
# 7. PRINCIPAL COORDINATES ANALYSIS (PCoA)
# =========================
dm = DistanceMatrix(bc_matrix.values, ids=samples_to_use)
pcoa_result = pcoa(dm)

pcoa_coords = pcoa_result.samples[["PC1", "PC2", "PC3"]].copy()
pcoa_coords.index.name = "Sample"
pcoa_coords = pcoa_coords.reset_index()
pcoa_coords = pcoa_coords.merge(sample_metadata, on="Sample")

print("\nPCoA coordinates:")
print(pcoa_coords)

pcoa_coords.to_csv(OUTPUT_PCOA, index=False)
print(f"\nSaved PCoA coordinates to: {OUTPUT_PCOA}")

explained = pcoa_result.proportion_explained
pc1_var = round(explained["PC1"] * 100, 1)
pc2_var = round(explained["PC2"] * 100, 1)
pc3_var = round(explained["PC3"] * 100, 1)

print(f"\nVariance explained — PC1: {pc1_var}%  |  PC2: {pc2_var}%  |  PC3: {pc3_var}%")

# =========================
# 8. PERMANOVA — runs BEFORE the plot so permanova_label is ready
# =========================
# PERMANOVA tests whether community composition differs significantly
# between groups.
#
# p-value   = probability of seeing this separation by chance
#             p < 0.05 means the groups are significantly different
#
# R-squared = proportion of total variation explained by group membership
#             must be between 0 and 1
#             e.g. R² = 0.45 means 45% of community variation
#             is explained by group (BULK / FSW / 2216)

group_labels = sample_metadata.set_index("Sample").loc[samples_to_use, "Group"].tolist()

permanova_result = permanova(dm, group_labels, permutations=N_PERMUTATIONS)

print("\nPERMANOVA result (all groups):")
print(permanova_result)

# ----------------------------------------------------------
# IMPORTANT: scikit-bio's "test statistic" field returns the
# pseudo-F statistic, NOT R². We calculate R² manually here.
#
# Formula:
#   R² = (F * df_groups) / (F * df_groups + df_residual)
#
# Where:
#   F           = pseudo-F statistic from PERMANOVA
#   df_groups   = number of groups - 1
#   df_residual = number of samples - number of groups
# ----------------------------------------------------------

n         = len(samples_to_use)
f_stat    = permanova_result["test statistic"]
df_groups = len(GROUPS_TO_KEEP) - 1
df_resid  = n - len(GROUPS_TO_KEEP)

permanova_rsq = round(
    (f_stat * df_groups) / (f_stat * df_groups + df_resid),
    3
)

permanova_pvalue = permanova_result["p-value"]

print(f"\nPseudo-F statistic : {f_stat:.3f}")
print(f"R² (calculated)    : {permanova_rsq}")
print(f"p-value            : {permanova_pvalue}")

# Format p-value for display on the plot
if permanova_pvalue < 0.001:
    pvalue_label = "p < 0.001"
elif permanova_pvalue <= 0.05:
    pvalue_label = f"p = {permanova_pvalue:.3f}"
else:
    pvalue_label = f"p = {permanova_pvalue:.3f} (n.s.)"

# This variable is used in the plot below
permanova_label = (
    f"PERMANOVA\n"
    f"R² = {permanova_rsq}\n"
    f"{pvalue_label}"
)

print(f"\nText for the plot:\n{permanova_label}")

# =========================
# 9. PAIRWISE PERMANOVA
# =========================
# Test each pair of groups separately.
# R² is calculated correctly for each pair as well.

pairwise_results = {}

pairs = [
    ("BULK", "FSW"),
    ("FSW",  "2216"),
    ("BULK", "2216")
]

for group_a, group_b in pairs:
    pair_samples = sample_metadata[
        sample_metadata["Group"].isin([group_a, group_b])
    ]["Sample"].tolist()

    pair_dm     = dm.filter(pair_samples)
    pair_labels = sample_metadata.set_index("Sample").loc[pair_samples, "Group"].tolist()

    result = permanova(pair_dm, pair_labels, permutations=N_PERMUTATIONS)

    # Correct R² for this pair
    n_pair       = len(pair_samples)
    f_pair       = result["test statistic"]
    df_grp_pair  = 1               # always 1 for a pairwise comparison
    df_res_pair  = n_pair - 2      # n samples - 2 groups

    rsq_pair = round(
        (f_pair * df_grp_pair) / (f_pair * df_grp_pair + df_res_pair),
        3
    )

    pairwise_results[f"{group_a} vs {group_b}"] = {
        "result":  result,
        "F":       round(f_pair, 3),
        "R2":      rsq_pair,
        "p-value": result["p-value"]
    }

    print(f"\nPERMANOVA: {group_a} vs {group_b}")
    print(f"  pseudo-F = {f_pair:.3f}")
    print(f"  R²       = {rsq_pair}")
    print(f"  p-value  = {result['p-value']}")

# =========================
# 10. SAVE STATS TO TEXT FILE
# =========================
with open(OUTPUT_STATS, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("PERMANOVA RESULTS — Naples 16S beta-diversity\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Distance metric : Bray-Curtis\n")
    f.write(f"Permutations    : {N_PERMUTATIONS}\n")
    f.write(f"Samples         : {samples_to_use}\n\n")

    f.write("-" * 60 + "\n")
    f.write("OVERALL PERMANOVA (BULK + FSW + 2216)\n")
    f.write("-" * 60 + "\n")
    f.write(str(permanova_result) + "\n")
    f.write(f"R² (calculated) : {permanova_rsq}\n\n")

    for comparison, data in pairwise_results.items():
        f.write("-" * 60 + "\n")
        f.write(f"PAIRWISE: {comparison}\n")
        f.write("-" * 60 + "\n")
        f.write(str(data["result"]) + "\n")
        f.write(f"R² (calculated) : {data['R2']}\n\n")

print(f"\nSaved PERMANOVA results to: {OUTPUT_STATS}")

# =========================
# 11. PLOT THE PCoA
# =========================
# BULK is shown at reduced opacity as a background reference point.
# FSW and 2216 are shown at full opacity as the primary comparison.
# This makes the FSW vs 2216 contrast the visual focus while
# still providing the ambient community context.
#
# A 95% confidence ellipse is drawn around each group to visually
# highlight the clustering of replicates and separation between groups.
# The ellipse is based on the covariance of PC1/PC2 coordinates within
# each group. n_std=2.0 corresponds approximately to 95% confidence.

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    Draw a confidence ellipse around a set of 2D points.
    n_std=2.0 corresponds approximately to a 95% confidence region.
    The ellipse shape is derived from the 2x2 covariance matrix of (x, y).
    """
    if len(x) < 2:
        return
    cov      = np.cov(x, y)
    pearson  = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse  = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x  = np.sqrt(cov[0, 0]) * n_std
    scale_y  = np.sqrt(cov[1, 1]) * n_std
    mean_x   = np.mean(x)
    mean_y   = np.mean(y)
    transf   = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

# Opacity settings: BULK de-emphasised, FSW and 2216 prominent
GROUP_ALPHA   = {"BULK": 0.25, "FSW": 1.0,  "2216": 1.0}
GROUP_ZORDER  = {"BULK": 2,    "FSW": 100,   "2216": 100}
ELLIPSE_ALPHA = {"BULK": 0.06, "FSW": 0.12,  "2216": 0.12}

fig, ax = plt.subplots(figsize=(15, 15))

# Draw filled confidence ellipses first (behind points)
for group in GROUPS_TO_KEEP:
    subset = pcoa_coords[pcoa_coords["Group"] == group]
    confidence_ellipse(
        subset["PC1"].values, subset["PC2"].values, ax,
        n_std=2.0,
        facecolor=GROUP_COLORS[group],
        edgecolor=GROUP_COLORS[group],
        alpha=ELLIPSE_ALPHA[group],
        linewidth=2, linestyle="-", zorder=1
    )

for group in GROUPS_TO_KEEP:
    subset = pcoa_coords[pcoa_coords["Group"] == group]

    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        label=group,
        color=GROUP_COLORS[group],
        marker=GROUP_MARKERS[group],
        s=800,
        alpha=GROUP_ALPHA[group],
        edgecolors="black",
        linewidths=2,
        zorder=GROUP_ZORDER[group]
    )

    for _, row in subset.iterrows():
        ax.annotate(
            row["Sample"],
            xy=(row["PC1"], row["PC2"]),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=30,
            color="gray" if group == "BULK" else "black",
            alpha=GROUP_ALPHA[group]
        )

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

ax.set_xlabel(f"PC1 ({pc1_var}% variance explained)", size=30)
ax.set_ylabel(f"PC2 ({pc2_var}% variance explained)", size=30)
ax.set_title("PCoA of Bray-Curtis dissimilarity", size=34)
plt.xticks(size=30)
plt.yticks(size=30)

ax.legend(loc="upper left", fontsize=30, borderaxespad=1)

# PERMANOVA text box — bottom left corner
ax.text(
    0.03, 0.03,
    permanova_label,
    transform=ax.transAxes,
    fontsize=20,
    verticalalignment="bottom",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.8
    )
)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved PCoA plot to: {OUTPUT_FIG}")

# =========================
# SCREE PLOT
# =========================
# Shows how much variance each PCoA axis explains.
# Answers: how many axes are needed to capture X% of total
# community variation?
# Uses bc_matrix already computed above — no extra input needed.

n_s    = len(samples_to_use)
D2     = bc_matrix.values ** 2
H_mat  = np.eye(n_s) - np.ones((n_s, n_s)) / n_s
G_mat  = -0.5 * H_mat @ D2 @ H_mat

eigenvalues_all, _ = np.linalg.eigh(G_mat)
eigenvalues_all    = np.sort(eigenvalues_all)[::-1]
pos_eig            = eigenvalues_all[eigenvalues_all > 0]
total_var          = pos_eig.sum()

pct_var    = pos_eig / total_var * 100
cumulative = np.cumsum(pct_var)

print("\n=== SCREE TABLE (Bacteria + Archaea) ===")
print(f"{'PC':<6} {'Variance %':>11} {'Cumulative %':>13}")
print("-" * 34)

thresholds_hit = {}
for i, (pct, cum) in enumerate(zip(pct_var, cumulative)):
    marker = ""
    for thresh in [50, 75, 90, 95]:
        if thresh not in thresholds_hit and cum >= thresh:
            thresholds_hit[thresh] = i + 1
            marker = f"  <- {thresh}%"
    print(f"PC{i+1:<4} {pct:>9.1f}%  {cum:>11.1f}%{marker}")

print("\nComponents needed to reach:")
for thresh, pc_num in sorted(thresholds_hit.items()):
    print(f"  {thresh}% variance -> {pc_num} PCs")

import matplotlib.patches as mpatches_scree

n_pcs_to_show = len(pos_eig)
pc_labels     = [f"PC{i+1}" for i in range(n_pcs_to_show)]

fig_scree, ax_scree = plt.subplots(figsize=(14, 9))

bars = ax_scree.bar(
    range(n_pcs_to_show), pct_var,
    color="steelblue", alpha=0.75, edgecolor="black", linewidth=0.6
)

ax2_scree = ax_scree.twinx()
ax2_scree.plot(
    range(n_pcs_to_show), cumulative,
    color="crimson", marker="o", markersize=6, linewidth=1.8
)
ax2_scree.set_ylabel("Cumulative variance explained (%)", size=22, color="crimson")
ax2_scree.tick_params(axis="y", labelcolor="crimson", labelsize=18)
ax2_scree.set_ylim(0, 110)
ax2_scree.grid(False)

for thresh, pc_num in thresholds_hit.items():
    ax2_scree.axhline(thresh, color="crimson", linewidth=0.8, linestyle=":", alpha=0.6)
    ax2_scree.text(n_pcs_to_show - 0.3, thresh + 1.5,
                   f"{thresh}%", fontsize=13, color="crimson", ha="right")

for i in range(2):
    bars[i].set_color("#2ca02c")
    bars[i].set_alpha(0.85)

green_patch = mpatches_scree.Patch(color="#2ca02c", alpha=0.85,
                                    label="PC1-PC2 (main PCoA figure)")
blue_patch  = mpatches_scree.Patch(color="steelblue", alpha=0.75,
                                    label="Remaining PCs")
red_line    = plt.Line2D([0],[0], color="crimson", marker="o",
                         markersize=6, linewidth=1.8,
                         label="Cumulative variance")

ax_scree.legend(handles=[green_patch, blue_patch, red_line],
                loc="center right", fontsize=16)
ax_scree.set_xticks(range(n_pcs_to_show))
ax_scree.set_xticklabels(pc_labels, size=18)
ax_scree.set_xlabel("Principal coordinate axis", size=22)
ax_scree.set_ylabel("Variance explained (%)", size=22)
ax_scree.set_title(
    "PCoA scree plot — Bray-Curtis dissimilarity\n"
    "(Bacteria + Archaea, contaminant genera removed)",
    size=22
)
ax_scree.tick_params(axis="y", labelsize=18)
ax_scree.set_ylim(0, max(pct_var) * 1.25)
ax_scree.grid(False)

plt.tight_layout()
plt.savefig("pcoa_scree_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("\nSaved scree plot to: pcoa_scree_plot.png")

# =========================
# 3D PCoA PLOT
# =========================
# Adds PC3 to show whether the third axis reveals additional
# structure beyond the PC1/PC2 projection.
# PC3 explains an additional ~8% variance (total ~68%).
#
# NOTE: best used as an exploratory/supplementary figure.
# The 2D PCoA remains the primary figure for the paper.

from mpl_toolkits.mplot3d import Axes3D

fig_3d = plt.figure(figsize=(16, 14))
ax_3d  = fig_3d.add_subplot(111, projection="3d")

GROUP_ALPHA_3D = {"BULK": 0.25, "FSW": 1.0, "2216": 1.0}

for group in GROUPS_TO_KEEP:
    subset = pcoa_coords[pcoa_coords["Group"] == group]

    ax_3d.scatter(
        subset["PC1"],
        subset["PC2"],
        subset["PC3"],
        label=group,
        color=GROUP_COLORS[group],
        marker=GROUP_MARKERS[group],
        s=400,
        alpha=GROUP_ALPHA_3D[group],
        edgecolors="black",
        linewidths=1.5,
        depthshade=False
    )

    for _, row in subset.iterrows():
        ax_3d.text(
            row["PC1"], row["PC2"], row["PC3"],
            f"  {row['Sample']}",
            fontsize=14,
            color="gray" if group == "BULK" else "black",
            alpha=GROUP_ALPHA_3D[group]
        )

ax_3d.set_xlabel(f"PC1 ({pc1_var}%)", fontsize=18, labelpad=10)
ax_3d.set_ylabel(f"PC2 ({pc2_var}%)", fontsize=18, labelpad=10)
ax_3d.set_zlabel(f"PC3 ({pc3_var}%)", fontsize=18, labelpad=10)
ax_3d.set_title(
    f"3D PCoA — Bray-Curtis dissimilarity\n"
    f"PC1+PC2+PC3 = {pc1_var + pc2_var + pc3_var:.1f}% variance explained",
    fontsize=20, pad=20
)
ax_3d.tick_params(labelsize=14)
ax_3d.legend(loc="upper left", fontsize=18)
ax_3d.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig("pcoa_3d_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved 3D PCoA plot to: pcoa_3d_plot.png")
print(f"\nVariance captured:")
print(f"  2D (PC1+PC2)    : {pc1_var + pc2_var:.1f}%")
print(f"  3D (PC1+PC2+PC3): {pc1_var + pc2_var + pc3_var:.1f}%")
print(f"  PC3 adds        : {pc3_var:.1f}% additional variance")

# =========================
# 12. QUICK INTERPRETATION GUIDE
# =========================
print("\n===== QUICK INTERPRETATION =====")
print(f"PC1 explains {pc1_var}% of variation.")
print(f"PC2 explains {pc2_var}% of variation.")
print()
print("In the PCoA plot:")
print("  - Points close together = similar bacterial communities")
print("  - Points far apart      = different communities")
print("  - Groups clustering separately = community composition differs")
print()
print("PERMANOVA shown on the plot:")
print(f"  {permanova_label}")
print()
print("Pairwise PERMANOVA summary:")
for comparison, data in pairwise_results.items():
    print(f"  {comparison}: R² = {data['R2']}, p = {data['p-value']}")
print()
print("Check permanova_results.txt for full details.")

# =========================
# 13. PERMDISP
# =========================
# PERMDISP tests whether the spread (dispersion) of samples
# within each group is similar across groups.
#
# Why this matters:
# PERMANOVA can detect significant results either because groups
# have genuinely different compositions, OR because one group is
# more internally variable than others. PERMDISP distinguishes
# between these two cases.
#
# How it works:
# - For each sample, it computes the distance from that sample
#   to the centroid of its group in multivariate space
# - It then tests whether these within-group distances differ
#   significantly between groups
# - A non-significant PERMDISP result (p > 0.05) means dispersion
#   is similar across groups, so the PERMANOVA result reflects
#   genuine compositional differences — the ideal outcome
#
# Output:
#   p-value > 0.05 → dispersion is homogeneous → PERMANOVA result
#                    is fully interpretable
#   p-value < 0.05 → dispersion differs between groups → should be
#                    acknowledged as a caveat alongside PERMANOVA

from skbio.stats.distance import permdisp

print("\n===== PERMDISP =====")

# Run PERMDISP on the same distance matrix and group labels used for PERMANOVA
permdisp_result = permdisp(dm, group_labels, permutations=N_PERMUTATIONS)

print("\nPERMDISP result (all groups):")
print(permdisp_result)

permdisp_pvalue = permdisp_result["p-value"]
permdisp_fstat  = round(permdisp_result["test statistic"], 3)

# Format for display
if permdisp_pvalue > 0.05:
    permdisp_interp = (
        "Dispersion is homogeneous across groups (p > 0.05).\n"
        "The PERMANOVA result reflects genuine compositional\n"
        "differences, not differences in within-group variability."
    )
else:
    permdisp_interp = (
        "Dispersion differs between groups (p < 0.05).\n"
        "Acknowledge this alongside the PERMANOVA result.\n"
        "It does not invalidate PERMANOVA but should be discussed."
    )

print(f"\nPERMDISP F = {permdisp_fstat}, p = {permdisp_pvalue}")
print(f"\nInterpretation:\n{permdisp_interp}")

# Save PERMDISP result to the same stats file
with open(OUTPUT_STATS, "a") as f:
    f.write("=" * 60 + "\n")
    f.write("PERMDISP RESULT\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(permdisp_result) + "\n\n")
    f.write(f"Interpretation:\n{permdisp_interp}\n")

print(f"\nPERMDISP result appended to: {OUTPUT_STATS}")

#%%
# -*- coding: utf-8 -*-
"""
STEP 9 of Naples dataset 16S analysis: Weighted UniFrac beta-diversity

Goal:
- Complement the Bray-Curtis beta-diversity analysis (Step 8) with a
  phylogenetically-informed distance metric
- Weighted UniFrac accounts for both read abundance AND evolutionary
  relationships between ZOTUs, giving a richer picture of community
  differences than Bray-Curtis alone
- The fact that UniFrac R² values are consistently higher than Bray-Curtis
  suggests that 2216 and FSW communities differ not only in composition
  but occupy deeply divergent branches of the bacterial phylogenetic tree

Key difference from Step 8:
  Bray-Curtis  = treats all ZOTUs as equally different
  Weighted UniFrac = ZOTUs that are more distantly related in the tree
                     contribute more to the dissimilarity score

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)
- p1112_run260227_16S_ZOTU_MSA.tre                  (MSA phylogenetic tree)

Output:
- weighted_unifrac_distance_matrix.csv
- unifrac_pcoa_coordinates.csv
- unifrac_permanova_results.txt
- unifrac_pcoa_plot.png

NOTE: scikit-bio must be installed:
  pip install scikit-bio
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skbio                  import TreeNode
from skbio.diversity        import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance   import DistanceMatrix, permanova, permdisp

# =========================
# 2. INPUT SETTINGS
# =========================
ZOTU_FILE  = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
TREE_FILE  = Path(r"C:\Users\Anelli\Desktop\Experiments\GDC_sequencing\Naples_dataset\p1112_run260227_16S_Results\e_OTU\p1112_run260227_16S_ZOTU_MSA.tre")

OUTPUT_DIST  = Path("weighted_unifrac_distance_matrix.csv")
OUTPUT_PCOA  = Path("unifrac_pcoa_coordinates.csv")
OUTPUT_STATS = Path("unifrac_permanova_results.txt")
OUTPUT_FIG   = Path("unifrac_pcoa_plot.png")

GROUPS_TO_KEEP = ["BULK", "FSW", "2216"]
GROUP_ORDER    = ["FSW", "BULK", "2216"]

GROUP_COLORS  = {"BULK": "tab:blue", "FSW": "tab:orange", "2216": "tab:green"}
GROUP_MARKERS = {"BULK": "o",        "FSW": "s",          "2216": "^"}

N_PERMUTATIONS = 9999

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(ZOTU_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [c for c in df.columns
               if c not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group(sample_name):
    if sample_name.startswith("2216-"):    return "2216"
    elif sample_name.startswith("FSW-C-"): return "FSW-C"
    elif sample_name.startswith("FSW-"):   return "FSW"
    elif sample_name.startswith("BULK-"):  return "BULK"
    elif sample_name.startswith("CTR-E-"): return "CTR-E"
    elif sample_name.startswith("CTR-PCR"):return "CTR-PCR"
    else: return "Unknown"

sample_metadata = pd.DataFrame({"Sample": sample_cols})
sample_metadata["Group"] = sample_metadata["Sample"].apply(assign_group)
sample_metadata = sample_metadata[
    sample_metadata["Group"].isin(GROUPS_TO_KEEP)
].copy().reset_index(drop=True)

print("\nSamples kept for UniFrac analysis:")
print(sample_metadata.to_string(index=False))

samples_to_use = sample_metadata["Sample"].tolist()

# =========================
# 5. BUILD RAW COUNT MATRIX
# =========================
# IMPORTANT: Weighted UniFrac requires RAW INTEGER COUNTS,
# not relative abundances. scikit-bio normalises internally.
# This is the key difference from the Bray-Curtis step.

count_matrix = df.set_index(zotu_col)[samples_to_use].copy()

print(f"\nCount matrix shape : {count_matrix.shape}  (ZOTUs × samples)")
print(f"Total reads        : {count_matrix.sum().sum():,}")

# =========================
# 6. LOAD AND ROOT THE TREE
# =========================
print(f"\nLoading phylogenetic tree: {TREE_FILE}")
tree = TreeNode.read(str(TREE_FILE), format="newick")
print(f"  Tips in tree : {sum(1 for _ in tree.tips())}")

# Match tree tips to ZOTUs in the count matrix
# ZOTUs not in the tree must be removed before UniFrac computation
tree_tips        = {tip.name for tip in tree.tips()}
zotus_in_data    = set(count_matrix.index)
missing_from_tree = zotus_in_data - tree_tips

if missing_from_tree:
    print(f"\nWARNING: {len(missing_from_tree)} ZOTUs not found in tree — removing them.")
    count_matrix = count_matrix.loc[count_matrix.index.isin(tree_tips)].copy()
else:
    print(f"All {len(zotus_in_data)} ZOTUs found in tree.")

# Root the tree at midpoint
# Weighted UniFrac requires a rooted tree.
# Midpoint rooting minimises the maximum distance from root to any tip.
print("Rooting tree at midpoint...")
tree = tree.root_at_midpoint()
print("Tree rooting complete.")

# =========================
# 7. COMPUTE WEIGHTED UNIFRAC
# =========================
print("\nComputing weighted UniFrac distances...")
print("(This may take 1-2 minutes with 2,000+ ZOTUs)")

# beta_diversity expects:
#   counts  = 2D array (samples × ZOTUs)  — note the transposition
#   ids     = sample names
#   taxa    = ZOTU IDs matching tree tip names
counts_array = count_matrix.T.values.astype(int)
zotu_ids     = count_matrix.index.tolist()

wu_dm = beta_diversity(
    metric="weighted_unifrac",
    counts=counts_array,
    ids=samples_to_use,
    taxa=zotu_ids,
    tree=tree,
    validate=True
)

print("Done.")

# Save distance matrix
wu_matrix = pd.DataFrame(
    wu_dm.data,
    index=samples_to_use,
    columns=samples_to_use
)
wu_matrix.to_csv(OUTPUT_DIST)
print(f"\nWeighted UniFrac distance matrix:")
print(wu_matrix.round(4).to_string())
print(f"\nSaved to: {OUTPUT_DIST}")

# Sanity check
samples_2216 = sample_metadata[sample_metadata["Group"]=="2216"]["Sample"].tolist()
samples_fsw  = sample_metadata[sample_metadata["Group"]=="FSW"]["Sample"].tolist()

within_2216  = [wu_matrix.loc[a,b] for i,a in enumerate(samples_2216)
                for b in samples_2216[i+1:]]
within_fsw   = [wu_matrix.loc[a,b] for i,a in enumerate(samples_fsw)
                for b in samples_fsw[i+1:]]
between      = [wu_matrix.loc[a,b] for a in samples_2216 for b in samples_fsw]

print(f"\nSanity check:")
print(f"  Mean within-2216   : {np.mean(within_2216):.4f}  (expect < between-group)")
print(f"  Mean within-FSW    : {np.mean(within_fsw):.4f}  (expect < between-group)")
print(f"  Mean 2216 vs FSW   : {np.mean(between):.4f}  (expect > within-group)")

# =========================
# 8. PCoA
# =========================
print("\nRunning PCoA...")

pcoa_result = pcoa(wu_dm)

pcoa_coords = pcoa_result.samples[["PC1", "PC2"]].copy()
pcoa_coords.index.name = "Sample"
pcoa_coords = pcoa_coords.reset_index()
pcoa_coords = pcoa_coords.merge(sample_metadata, on="Sample")

pcoa_coords.to_csv(OUTPUT_PCOA, index=False)
print(f"Saved PCoA coordinates to: {OUTPUT_PCOA}")

explained = pcoa_result.proportion_explained
pc1_var   = round(explained["PC1"] * 100, 1)
pc2_var   = round(explained["PC2"] * 100, 1)
print(f"\nVariance explained — PC1: {pc1_var}%  |  PC2: {pc2_var}%")

# =========================
# 9. PERMANOVA — runs BEFORE plot so label is ready
# =========================
group_labels = sample_metadata.set_index("Sample").loc[
    samples_to_use, "Group"
].tolist()

permanova_result = permanova(wu_dm, group_labels, permutations=N_PERMUTATIONS)

print("\nPERMANOVA result (all groups):")
print(permanova_result)

# Compute R² from pseudo-F
n         = len(samples_to_use)
f_stat    = permanova_result["test statistic"]
df_groups = len(GROUPS_TO_KEEP) - 1
df_resid  = n - len(GROUPS_TO_KEEP)

permanova_rsq  = round((f_stat * df_groups) / (f_stat * df_groups + df_resid), 3)
permanova_pval = permanova_result["p-value"]

print(f"\nPseudo-F : {f_stat:.3f}")
print(f"R²       : {permanova_rsq}")
print(f"p-value  : {permanova_pval}")

if permanova_pval < 0.001:
    pval_str = "p < 0.001"
elif permanova_pval <= 0.05:
    pval_str = f"p = {permanova_pval:.3f}"
else:
    pval_str = f"p = {permanova_pval:.3f} (n.s.)"

permanova_label_unifrac = (
    f"PERMANOVA\n"
    f"R² = {permanova_rsq}\n"
    f"{pval_str}"
)

# =========================
# 10. PAIRWISE PERMANOVA
# =========================
pairwise_results = {}
pairs = [("BULK","FSW"), ("FSW","2216"), ("BULK","2216")]

for group_a, group_b in pairs:
    pair_samples = sample_metadata[
        sample_metadata["Group"].isin([group_a, group_b])
    ]["Sample"].tolist()

    pair_dm     = wu_dm.filter(pair_samples)
    pair_labels = sample_metadata.set_index("Sample").loc[
        pair_samples, "Group"
    ].tolist()

    result   = permanova(pair_dm, pair_labels, permutations=N_PERMUTATIONS)
    n_pair   = len(pair_samples)
    f_pair   = result["test statistic"]
    rsq_pair = round((f_pair) / (f_pair + (n_pair - 2)), 3)

    pairwise_results[f"{group_a} vs {group_b}"] = {
        "result":  result,
        "R2":      rsq_pair,
        "p-value": result["p-value"]
    }

    print(f"\nPERMANOVA: {group_a} vs {group_b}")
    print(f"  pseudo-F = {f_pair:.3f}")
    print(f"  R²       = {rsq_pair}")
    print(f"  p-value  = {result['p-value']}")

# =========================
# 11. PERMDISP
# =========================
permdisp_result = permdisp(wu_dm, group_labels, permutations=N_PERMUTATIONS)

print("\nPERMDISP result:")
print(permdisp_result)

permdisp_fstat = round(permdisp_result["test statistic"], 3)
permdisp_pval  = permdisp_result["p-value"]

if permdisp_pval > 0.05:
    permdisp_interp = (
        "Dispersion is homogeneous across groups (p > 0.05).\n"
        "The PERMANOVA result reflects genuine compositional\n"
        "differences, not differences in within-group variability."
    )
else:
    permdisp_interp = (
        "Dispersion differs between groups (p < 0.05).\n"
        "Acknowledge this alongside the PERMANOVA result.\n"
        "It does not invalidate PERMANOVA but should be discussed."
    )

print(f"\nPERMDISP F = {permdisp_fstat}, p = {permdisp_pval}")
print(f"\nInterpretation:\n{permdisp_interp}")

# =========================
# 12. SAVE STATISTICS
# =========================
with open(OUTPUT_STATS, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("WEIGHTED UNIFRAC — PERMANOVA RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Distance metric : Weighted UniFrac\n")
    f.write(f"Tree            : {TREE_FILE.name}\n")
    f.write(f"Permutations    : {N_PERMUTATIONS}\n")
    f.write(f"Samples         : {samples_to_use}\n\n")

    f.write("-" * 60 + "\n")
    f.write("OVERALL PERMANOVA (BULK + FSW + 2216)\n")
    f.write("-" * 60 + "\n")
    f.write(str(permanova_result) + "\n")
    f.write(f"R² (calculated) : {permanova_rsq}\n\n")

    for comparison, data in pairwise_results.items():
        f.write("-" * 60 + "\n")
        f.write(f"PAIRWISE: {comparison}\n")
        f.write("-" * 60 + "\n")
        f.write(str(data["result"]) + "\n")
        f.write(f"R² (calculated) : {data['R2']}\n\n")

    f.write("=" * 60 + "\n")
    f.write("PERMDISP RESULT\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(permdisp_result) + "\n\n")
    f.write(f"Interpretation:\n{permdisp_interp}\n")

print(f"\nSaved statistics to: {OUTPUT_STATS}")

# =========================
# 13. PLOT — PCoA (classic style, consistent with Step 8)
# =========================
# BULK is shown at reduced opacity as a background reference point.
# FSW and 2216 are shown at full opacity as the primary comparison.
# Confidence ellipses (95%) consistent with Step 8.

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse_wu(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """Draw a 95% confidence ellipse. See Step 8 for full documentation."""
    if len(x) < 2:
        return
    cov      = np.cov(x, y)
    pearson  = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse  = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x  = np.sqrt(cov[0, 0]) * n_std
    scale_y  = np.sqrt(cov[1, 1]) * n_std
    mean_x   = np.mean(x)
    mean_y   = np.mean(y)
    transf   = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

GROUP_ALPHA_WU   = {"BULK": 0.25, "FSW": 1.0,  "2216": 1.0}
GROUP_ZORDER_WU  = {"BULK": 2,    "FSW": 100,   "2216": 100}
ELLIPSE_ALPHA_WU = {"BULK": 0.06, "FSW": 0.12,  "2216": 0.12}

fig, ax = plt.subplots(figsize=(15, 15))

# Draw filled confidence ellipses first (behind points)
for group in GROUPS_TO_KEEP:
    subset = pcoa_coords[pcoa_coords["Group"] == group]
    confidence_ellipse_wu(
        subset["PC1"].values, subset["PC2"].values, ax,
        n_std=2.0,
        facecolor=GROUP_COLORS[group],
        edgecolor=GROUP_COLORS[group],
        alpha=ELLIPSE_ALPHA_WU[group],
        linewidth=2, linestyle="-", zorder=1
    )

for group in GROUPS_TO_KEEP:
    subset = pcoa_coords[pcoa_coords["Group"] == group]

    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        label=group,
        color=GROUP_COLORS[group],
        marker=GROUP_MARKERS[group],
        s=500,
        alpha=GROUP_ALPHA_WU[group],
        edgecolors="black",
        linewidths=2,
        zorder=GROUP_ZORDER_WU[group]
    )

    for _, row in subset.iterrows():
        ax.annotate(
            row["Sample"],
            xy=(row["PC1"], row["PC2"]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=20,
            color="gray" if group == "BULK" else "black",
            alpha=GROUP_ALPHA_WU[group]
        )

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

ax.set_xlabel(f"PC1 ({pc1_var}% variance explained)", size=30)
ax.set_ylabel(f"PC2 ({pc2_var}% variance explained)", size=30)
ax.set_title("PCoA of Weighted UniFrac distance", size=34)
plt.xticks(size=30)
plt.yticks(size=30)

ax.legend(loc="upper right", fontsize=30, borderaxespad=1)

# PERMANOVA text box — bottom left corner
ax.text(
    0.03, 0.03,
    permanova_label_unifrac,
    transform=ax.transAxes,
    fontsize=20,
    verticalalignment="bottom",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.8
    )
)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved PCoA plot to: {OUTPUT_FIG}")

# =========================
# 14. INTERPRETATION GUIDE
# =========================
print("\n===== INTERPRETATION =====")
print()
print("Compare these results with the Bray-Curtis analysis (Step 8).")
print()
print("If weighted UniFrac R² > Bray-Curtis R², it means the communities")
print("differ not just in composition but in deep phylogenetic structure —")
print("they occupy distinct branches of the tree of life.")
print()
print(f"Weighted UniFrac PERMANOVA : R² = {permanova_rsq}, {pval_str}")
print()
print("Pairwise summary:")
for comparison, data in pairwise_results.items():
    print(f"  {comparison}: R² = {data['R2']}, p = {data['p-value']}")
print()
print(f"PERMDISP: F = {permdisp_fstat}, p = {permdisp_pval}")
print(permdisp_interp)
#%%
# -*- coding: utf-8 -*-
"""
STEP 10 of Naples dataset 16S analysis: Family-level heatmap with hierarchical clustering

Goal:
- Visualise bacterial community composition at family level across all
  individual samples (not pooled by group)
- Hierarchical clustering of samples (columns) reveals which samples
  are most similar to each other and whether group identity drives
  the clustering pattern
- Complements the stacked bar plot (Step 4) which shows pooled group profiles
- Inspired by Lambert et al. (2017, Nature Microbiology) Supp. Fig. 12,
  the original ISCA paper

Design choices:
- Top 20 families by mean relative abundance in 2216 samples
  (so chemotaxis-enriched taxa appear at the top of the heatmap)
- Rows = families, sorted by mean abundance in 2216 (most enriched at top)
- Columns = individual samples, hierarchically clustered by Ward/Bray-Curtis
- Colour scale = blue-white-red diverging, centred at mean log abundance
- Log10 transformation applied so rare taxa are visible alongside dominant ones
- Group annotation bar above columns (Wong palette: FSW/BULK/2216)

Input:
- filtered_zotu_table_bacteria_no_contaminants.txt  (output of Step 1)

Output:
- heatmap_family_clustered.png
"""

# =========================
# 1. IMPORT PACKAGES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

# =========================
# 2. INPUT SETTINGS
# =========================
INPUT_FILE     = Path("filtered_zotu_table_bacteria_no_contaminants.txt")
OUTPUT_FIG     = Path("heatmap_family_clustered.png")

TOP_N_FAMILIES = 25
GROUPS_TO_KEEP = ["FSW", "BULK", "2216"]

# Wong colour palette (consistent with all other steps)
GROUP_COLORS = {
    "BULK": "#0072B2",
    "FSW":  "#E69F00",
    "2216": "#009E73"
}

# =========================
# 3. READ FILTERED TABLE
# =========================
df = pd.read_csv(INPUT_FILE, sep="\t")

zotu_col     = "#OTU ID"
taxonomy_col = "Consensus Lineage"
genus_col    = "Genus"

sample_cols = [c for c in df.columns
               if c not in [zotu_col, taxonomy_col, genus_col]]

print("Detected sample columns:")
print(sample_cols)

# =========================
# 4. ASSIGN SAMPLE GROUPS
# =========================
def assign_group_hm(s):
    if s.startswith("2216-"):    return "2216"
    elif s.startswith("FSW-C-"): return "FSW-C"
    elif s.startswith("FSW-"):   return "FSW"
    elif s.startswith("BULK-"):  return "BULK"
    else:                        return "Unknown"

sample_metadata_hm = pd.DataFrame({"Sample": sample_cols})
sample_metadata_hm["Group"] = sample_metadata_hm["Sample"].apply(assign_group_hm)
sample_metadata_hm = sample_metadata_hm[
    sample_metadata_hm["Group"].isin(GROUPS_TO_KEEP)
].copy().reset_index(drop=True)

samples_to_use_hm = sample_metadata_hm["Sample"].tolist()
print(f"\nSamples used: {samples_to_use_hm}")

# =========================
# 5. EXTRACT FAMILY FROM TAXONOMY
# =========================
def extract_family_hm(t):
    """
    Extract the most specific available taxonomic label from a SILVA
    taxonomy string, climbing up the hierarchy until something
    informative is found.

    Priority order:
      family (f__) → order (o__) → class (c__) → phylum (p__)

    Each fallback level is labelled with a suffix so the reader
    knows the resolution, e.g. "Rhizobiales (order)" or
    "Alphaproteobacteria (class)".

    Only returns "Unclassified" if no level above family has a
    meaningful name — which is extremely rare in SILVA 138.
    """
    if pd.isna(t): return "Unclassified"

    parts = [p.strip() for p in str(t).split(";") if p.strip()]
    bad   = {"uncultured", "unclassified", "unknown", "metagenome", ""}

    # 1. Try family
    for p in parts:
        if p.startswith("f__"):
            val = p.replace("f__", "").strip()
            if val.lower() not in bad:
                return val

    # 2. Try order
    for p in parts:
        if p.startswith("o__"):
            val = p.replace("o__", "").strip()
            if val.lower() not in bad:
                return f"{val} (order)"

    # 3. Try class
    for p in parts:
        if p.startswith("c__"):
            val = p.replace("c__", "").strip()
            if val.lower() not in bad:
                return f"{val} (class)"

    # 4. Try phylum
    for p in parts:
        if p.startswith("p__"):
            val = p.replace("p__", "").strip()
            if val.lower() not in bad:
                return f"{val} (phylum)"

    return "Unclassified"

df["Family_hm"] = df[taxonomy_col].apply(extract_family_hm)

# =========================
# 6. COMPUTE PER-SAMPLE RELATIVE ABUNDANCE AT FAMILY LEVEL
# =========================
# Each sample is kept separate — no pooling.
# This preserves replicate-level variation so the clustering
# reflects genuine within-group consistency.

rel_hm = df[samples_to_use_hm].copy()
rel_hm = rel_hm.div(rel_hm.sum(axis=0), axis=1)
rel_hm["Family_hm"] = df["Family_hm"].values

family_rel_hm = rel_hm.groupby("Family_hm")[samples_to_use_hm].sum()

print(f"\nFamily-level matrix: {family_rel_hm.shape}  (families x samples)")

# =========================
# 7. SELECT TOP 20 FAMILIES
# =========================
# Ranked by mean relative abundance in 2216 samples so the most
# chemotaxis-enriched families appear at the top of the heatmap.

samples_2216_hm = sample_metadata_hm[
    sample_metadata_hm["Group"] == "2216"
]["Sample"].tolist()

family_rel_hm["Mean_2216"] = family_rel_hm[samples_2216_hm].mean(axis=1)

top_families_hm = (
    family_rel_hm
    .sort_values("Mean_2216", ascending=False)
    .head(TOP_N_FAMILIES)
    .drop(columns="Mean_2216")
)

print(f"\nTop {TOP_N_FAMILIES} families (sorted by mean abundance in 2216):")
for fam in top_families_hm.index:
    print(f"  {fam}")

# =========================
# 8. LOG10 TRANSFORM
# =========================
# Log10 makes rare taxa visible alongside dominant ones.
# Without this, Marinimicrobia (~13%) would dominate the colour scale
# and low-abundance families would be invisible.
# A pseudocount of 1e-5 avoids log(0).

PSEUDOCOUNT  = 1e-5
log_matrix_hm = np.log10(top_families_hm + PSEUDOCOUNT)

print(f"\nLog10 matrix range: {log_matrix_hm.values.min():.2f} to {log_matrix_hm.values.max():.2f}")

# =========================
# 9. HIERARCHICAL CLUSTERING OF SAMPLES (COLUMNS ONLY)
# =========================
# Bray-Curtis dissimilarity on the top-20 family matrix,
# then Ward linkage to produce compact, interpretable clusters.
# Rows (families) are kept in fixed order (sorted by 2216 abundance).

bc_dist_hm  = pdist(top_families_hm.T.values, metric="braycurtis")
Z_hm        = linkage(bc_dist_hm, method="ward")
col_order_hm = leaves_list(Z_hm)

ordered_samples_hm = [samples_to_use_hm[i] for i in col_order_hm]
ordered_groups_hm  = [
    sample_metadata_hm.set_index("Sample").loc[s, "Group"]
    for s in ordered_samples_hm
]

log_ordered_hm = log_matrix_hm[ordered_samples_hm]

print(f"\nSample order after hierarchical clustering:")
for s, g in zip(ordered_samples_hm, ordered_groups_hm):
    print(f"  {s}  ({g})")

# =========================
# 10. PLOT
# =========================
# Layout (3 rows):
#   Row 0 (thin)  : group colour annotation bar
#   Row 1 (medium): dendrogram
#   Row 2 (tall)  : heatmap
# Plus a narrow colourbar column on the right.

fig_hm = plt.figure(figsize=(28, 22))

gs_hm = gridspec.GridSpec(
    3, 2,
    height_ratios=[0.04, 0.15, 1],
    width_ratios=[1, 0.04],
    hspace=0.02,
    wspace=0.03,
    left=0.30      # reserves 30% of figure width for y-axis labels
)

ax_annot_hm   = fig_hm.add_subplot(gs_hm[0, 0])
ax_dendro_hm  = fig_hm.add_subplot(gs_hm[1, 0])
ax_heatmap_hm = fig_hm.add_subplot(gs_hm[2, 0])
ax_cbar_hm    = fig_hm.add_subplot(gs_hm[2, 1])

# ── Group annotation bar ─────────────────────────────────────
for i, (sample, group) in enumerate(zip(ordered_samples_hm, ordered_groups_hm)):
    ax_annot_hm.add_patch(plt.Rectangle(
        (i, 0), 1, 1,
        color=GROUP_COLORS[group]
    ))

ax_annot_hm.set_xlim(0, len(ordered_samples_hm))
ax_annot_hm.set_ylim(0, 1)
ax_annot_hm.axis("off")

legend_patches_hm = [
    mpatches.Patch(color=GROUP_COLORS[g], label=g)
    for g in GROUPS_TO_KEEP
]
ax_annot_hm.legend(
    handles=legend_patches_hm,
    loc="upper left",
    bbox_to_anchor=(0, 3.0),
    ncol=3,
    fontsize=28,
    frameon=False
)

# ── Dendrogram ────────────────────────────────────────────────
dendrogram(
    Z_hm,
    ax=ax_dendro_hm,
    color_threshold=0,
    above_threshold_color="black",
    link_color_func=lambda k: "black"
)
ax_dendro_hm.axis("off")

# ── Heatmap ───────────────────────────────────────────────────
# Diverging colour scale centred at the mean log value.
# Blue = below average abundance, red = above average abundance.
vmin_hm    = log_ordered_hm.values.min()
vmax_hm    = log_ordered_hm.values.max()
vcenter_hm = log_ordered_hm.values.mean()

norm_hm = TwoSlopeNorm(vmin=vmin_hm, vcenter=vcenter_hm, vmax=vmax_hm)
cmap_hm = plt.cm.RdBu_r   # blue=low, white=centre, red=high

im_hm = ax_heatmap_hm.imshow(
    log_ordered_hm.values,
    aspect="auto",
    cmap=cmap_hm,
    norm=norm_hm,
    interpolation="nearest"
)

# Axis tick labels
ax_heatmap_hm.set_xticks(range(len(ordered_samples_hm)))
ax_heatmap_hm.set_xticklabels(
    ordered_samples_hm,
    rotation=45, ha="right", fontsize=24
)

ax_heatmap_hm.set_yticks(range(TOP_N_FAMILIES))
ax_heatmap_hm.set_yticklabels(log_ordered_hm.index, fontsize=24)

ax_heatmap_hm.set_ylabel("Family", fontsize=24)

# Subtle white grid lines between cells
ax_heatmap_hm.set_xticks(np.arange(-0.5, len(ordered_samples_hm), 1), minor=True)
ax_heatmap_hm.set_yticks(np.arange(-0.5, TOP_N_FAMILIES, 1), minor=True)
ax_heatmap_hm.grid(which="minor", color="white", linewidth=0.5)
ax_heatmap_hm.tick_params(which="minor", bottom=False, left=False)

# ── Colourbar ─────────────────────────────────────────────────
cbar_hm = fig_hm.colorbar(im_hm, cax=ax_cbar_hm)
cbar_hm.set_label(
    "log₁₀ (relative abundance)",
    fontsize=24, rotation=270, labelpad=24
)
cbar_hm.ax.tick_params(labelsize=24)

plt.subplots_adjust(left=0.35)
fig_hm.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved heatmap to: {OUTPUT_FIG}")

# =========================
# 11. INTERPRETATION GUIDE
# =========================
print("\n===== INTERPRETATION =====")
print()
print("Rows    = top 20 families, sorted by mean abundance in 2216 (top = most enriched)")
print("Columns = individual samples, ordered by hierarchical clustering")
print("Colour  = log10(relative abundance)")
print("          Red   = above average abundance")
print("          Blue  = below average abundance")
print("          White = mean abundance")
print()
print("What to look for:")
print("  - Do 2216 samples cluster together?")
print("    -> Confirms consistent enrichment across replicates")
print("  - Do FSW samples cluster together?")
print("    -> Confirms consistent passive-entry community across replicates")
print("  - Which families are red in 2216 and blue in FSW?")
print("    -> These are your chemoattractant-selected taxa")
print("  - Which families are red in FSW and blue in 2216?")
print("    -> These are passive-entry associated taxa")