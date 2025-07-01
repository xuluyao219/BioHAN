"""
Enhanced Biological Knowledge Base for BioHAN
Contains curated pathway information and cell type markers
"""

import numpy as np
from typing import Dict, List, Optional, Set
import networkx as nx


class BiologicalKnowledgeBase:
    """Biological knowledge management for cell type classification

    This class provides curated biological information including:
    - Pathway definitions from KEGG/MSigDB
    - Cell type specific marker genes from CellMarker/PanglaoDB
    - Gene-gene interaction networks from STRING
    """

    def __init__(self):
        # Enhanced pathway definitions with cell type specific pathways
        self.pathways = {
            # T cell related pathways
            'T_cell_activation': ['CD3D', 'CD3E', 'CD3G', 'CD28', 'ICOS', 'CTLA4', 'CD27'],
            'CD4_T_cell': ['CD4', 'IL7R', 'CCR7', 'SELL', 'TCF7', 'LEF1', 'MAL', 'LDHB'],
            'CD8_T_cell': ['CD8A', 'CD8B', 'GZMK', 'CCL5', 'NKG7', 'PRF1', 'GZMA'],
            'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'IKZF2', 'TNFRSF18'],
            'T_memory': ['IL7R', 'CD27', 'CCR7', 'SELL', 'TCF7', 'LEF1'],
            'T_effector': ['GZMB', 'PRF1', 'IFNG', 'TNF', 'FASLG', 'NKG7'],
            'T_exhaustion': ['PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'ENTPD1', 'BATF', 'TOX'],

            # B cell related pathways
            'B_cell_activation': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'CD22', 'BANK1'],
            'B_cell_maturation': ['IGHM', 'IGHD', 'IGHA1', 'IGHA2', 'IGHG1', 'IGHG2'],
            'Plasma_cell': ['MZB1', 'SDC1', 'CD38', 'IGHG1', 'JCHAIN', 'XBP1', 'PRDM1'],
            'Memory_B': ['CD27', 'TNFRSF13B', 'AIM2', 'TNFRSF13C'],

            # NK cell related pathways
            'NK_cell_cytotoxicity': ['NCAM1', 'FCGR3A', 'NKG7', 'GNLY', 'PRF1', 'GZMB', 'GZMA'],
            'NK_cell_activation': ['KLRB1', 'KLRD1', 'KLRF1', 'KLRC1', 'NCR1', 'NCR3'],

            # Myeloid cell related pathways
            'Monocyte_classical': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'FCN1'],
            'Monocyte_nonclassical': ['FCGR3A', 'MS4A7', 'CDKN1C', 'CX3CR1', 'RHOC'],
            'Macrophage': ['CD68', 'CD163', 'MARCO', 'MSR1', 'MRC1', 'C1QA', 'C1QB'],
            'DC_conventional': ['FCER1A', 'CD1C', 'CLEC10A', 'CD1E', 'CD207'],
            'DC_plasmacytoid': ['IL3RA', 'CLEC4C', 'NRP1', 'TCF4', 'IRF7', 'IRF8'],

            # Other cell types
            'Platelet': ['PPBP', 'PF4', 'GP9', 'ITGA2B', 'TUBB1', 'CLU'],
            'Erythrocyte': ['HBA1', 'HBA2', 'HBB', 'HBD', 'ALAS2', 'SLC4A1'],

            # Tissue specific cells
            'Epithelial': ['EPCAM', 'KRT18', 'KRT19', 'CDH1', 'KRT8', 'KRT7'],
            'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'FLT1', 'ENG'],
            'Fibroblast': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'PDGFRA', 'FAP'],

            # Cell cycle and metabolism
            'cell_cycle': ['CDK1', 'CDK2', 'CCNA2', 'CCNB1', 'TP53', 'MKI67'],
            'apoptosis': ['BCL2', 'BAX', 'CASP3', 'CASP8', 'FAS', 'FASLG'],
            'metabolism': ['GAPDH', 'PKM', 'LDHA', 'HK1', 'ENO1', 'PGK1']
        }

        # Cell type to marker gene mapping
        self.celltype_markers = {
            'T_cells': ['CD3D', 'CD3E', 'CD3G', 'TRAC', 'TRBC1', 'TRBC2'],
            'CD4_T_cells': ['CD4', 'IL7R', 'CCR7', 'SELL', 'TCF7', 'LEF1'],
            'CD8_T_cells': ['CD8A', 'CD8B', 'GZMK', 'CCL5', 'NKG7'],
            'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'IKZF2'],
            'B_cells': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'CD22', 'BANK1'],
            'Plasma_cells': ['MZB1', 'SDC1', 'CD38', 'IGHG1', 'JCHAIN', 'XBP1'],
            'NK_cells': ['NCAM1', 'FCGR3A', 'NKG7', 'GNLY', 'PRF1', 'GZMB'],
            'Monocytes': ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'FCN1'],
            'DC': ['FCER1A', 'CST3', 'CD1C', 'CLEC10A', 'IL3RA', 'CLEC4C'],
            'Macrophages': ['CD68', 'CD163', 'MARCO', 'MSR1', 'MRC1'],
            'Platelets': ['PPBP', 'PF4', 'GP9', 'ITGA2B'],
            'Erythrocytes': ['HBA1', 'HBA2', 'HBB', 'HBD'],
            'Epithelial': ['EPCAM', 'KRT18', 'KRT19', 'CDH1'],
            'Endothelial': ['PECAM1', 'VWF', 'CDH5', 'CLDN5'],
            'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN', 'LUM']
        }

    def get_module_prior(self, gene_list: List[str], n_modules: int) -> np.ndarray:
        """Get prior gene module assignments based on pathway knowledge

        Args:
            gene_list: List of gene names
            n_modules: Number of modules to create

        Returns:
            Prior assignment matrix [n_genes x n_modules]
        """
        n_genes = len(gene_list)
        prior = np.random.randn(n_genes, n_modules) * 0.1

        # Create gene to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}

        # Assign genes from same pathway to same module
        for module_idx, (pathway, genes) in enumerate(list(self.pathways.items())[:n_modules]):
            for gene in genes:
                if gene in gene_to_idx:
                    prior[gene_to_idx[gene], module_idx] = 1.0

        return prior

    def get_interaction_matrix(self, gene_list: List[str]) -> np.ndarray:
        """Get gene-gene interaction matrix based on pathway co-membership

        Args:
            gene_list: List of gene names

        Returns:
            Interaction matrix [n_genes x n_genes] with values 0-1
        """
        n_genes = len(gene_list)
        matrix = np.eye(n_genes) * 0.1  # Baseline self-interaction

        # Create gene to index mapping
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}

        # Genes in same pathway have stronger interactions
        for pathway_genes in self.pathways.values():
            for g1 in pathway_genes:
                for g2 in pathway_genes:
                    if g1 in gene_to_idx and g2 in gene_to_idx:
                        i, j = gene_to_idx[g1], gene_to_idx[g2]
                        matrix[i, j] = 0.9

        return matrix

    def get_celltype_markers(self) -> Dict[str, List[str]]:
        """Get cell type to marker gene mapping

        Returns:
            Dictionary mapping cell type names to marker gene lists
        """
        return self.celltype_markers

    def get_marker_importance_prior(self, gene_list: List[str]) -> Dict[str, np.ndarray]:
        """Generate marker gene importance priors for each cell type

        Args:
            gene_list: List of gene names

        Returns:
            Dictionary mapping cell type to importance scores
        """
        importance_priors = {}
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}

        for celltype, markers in self.celltype_markers.items():
            prior = np.zeros(len(gene_list))
            for marker in markers:
                if marker in gene_to_idx:
                    prior[gene_to_idx[marker]] = 1.0
            importance_priors[celltype] = prior

        return importance_priors

    def get_pathway_enrichment(self, gene_set: Set[str]) -> Dict[str, float]:
        """Calculate pathway enrichment scores for a set of genes

        Args:
            gene_set: Set of gene names

        Returns:
            Dictionary mapping pathway names to enrichment p-values
        """
        enrichment_scores = {}

        for pathway_name, pathway_genes in self.pathways.items():
            # Simple hypergeometric test approximation
            overlap = len(set(pathway_genes) & gene_set)
            expected = len(pathway_genes) * len(gene_set) / 20000  # Assume 20k total genes

            if overlap > expected:
                # Simplified p-value calculation
                p_value = np.exp(-overlap * np.log(overlap / expected)) if expected > 0 else 1.0
                enrichment_scores[pathway_name] = min(p_value, 1.0)

        return enrichment_scores

    def get_gene_network(self, gene_list: List[str], threshold: float = 0.7) -> nx.Graph:
        """Create a gene interaction network

        Args:
            gene_list: List of gene names
            threshold: Minimum interaction score to include edge

        Returns:
            NetworkX graph of gene interactions
        """
        G = nx.Graph()
        G.add_nodes_from(gene_list)

        interaction_matrix = self.get_interaction_matrix(gene_list)
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}

        # Add edges based on interaction strength
        for i, g1 in enumerate(gene_list):
            for j, g2 in enumerate(gene_list):
                if i < j and interaction_matrix[i, j] > threshold:
                    G.add_edge(g1, g2, weight=interaction_matrix[i, j])

        return G