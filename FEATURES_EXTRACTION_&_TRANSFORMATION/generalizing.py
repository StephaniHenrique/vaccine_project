import pandas as pd

def classify_intermediate(row_str):
    if not isinstance(row_str, str):
        return "Unknown"
    
    markers = [m.strip() for m in row_str.split(',')]
    markers_set = set(markers)
    
    if 'pDC' in markers_set:
        return 'Plasmacytoid DCs (pDC)'
    elif 'mDC1' in markers_set:
        return 'Myeloid DCs type 1 (mDC1)'
    elif 'mDC2' in markers_set:
        return 'Myeloid DCs type 2 (mDC2)'
    elif 'mDC' in markers_set:
        return 'Myeloid DCs (Unspecified)'
    elif 'CD11c+' in markers_set and 'HLA-DR+' in markers_set and 'Lin1-' in markers_set:
        return 'Myeloid DCs (Unspecified)'
    elif 'HLA-DR+' in markers_set and 'Lin1-' in markers_set:
        return 'Dendritic Cells (Pan-DC)'
        
    elif 'B_naive' in markers_set or ('naive' in markers_set and 'CD19+' in markers_set):
        return 'Naive B cells'
    elif any(m in markers_set for m in ['B_switched_memory', 'B_unswitched_memory']):
        return 'Memory B cells'
    elif 'plasma_cell' in markers_set or 'plasmablast' in markers_set:
        return 'Plasmablasts / Plasma cells'
    elif 'CD19+' in markers_set or 'CD20+' in markers_set:
        return 'B cells (Other)'
        
    elif 'monocyte' in markers_set or 'CD14+' in markers_set or 'SLAN+' in markers_set or ('CD14-' in markers_set and 'CD16+' in markers_set):
        if 'CD14+' in markers_set and 'CD16-' in markers_set:
            return 'Classical Monocytes'
        elif 'CD14+' in markers_set and 'CD16+' in markers_set:
            return 'Intermediate Monocytes'
        elif 'CD14-' in markers_set and 'CD16+' in markers_set:
            return 'Non-classical Monocytes'
        elif 'SLAN+' in markers_set:
            return 'Non-classical Monocytes'
        return 'Monocytes (Unspecified)'
        
    elif any(m in markers_set for m in ['TFH', 'TFH1', 'TFH2', 'TFH17', 'TFH1-17']):
        return 'T follicular helper (Tfh) cells'
    elif any(m in markers_set for m in ['TH1', 'TH2', 'TH17', 'TH1-17', 'TH']):
        return 'CD4+ Helper T cells (Th subsets)'
    elif 'FoxP3+' in markers_set:
        return 'Regulatory T cells (Treg)'
    elif 'CD4+' in markers_set:
        if 'naive' in markers_set:
            return 'Naive CD4+ T cells'
        elif any(m in markers_set for m in ['CM', 'EM', 'TEMRA']):
            return 'Memory CD4+ T cells'
        return 'CD4+ T cells (Other)'
    elif 'CD8+' in markers_set:
        if 'naive' in markers_set:
            return 'Naive CD8+ T cells'
        elif any(m in markers_set for m in ['CM', 'EM', 'TEMRA']):
            return 'Memory CD8+ T cells'
        return 'CD8+ T cells (Other)'
    elif 'CD3+' in markers_set:
        return 'T cells (Unspecified Subtype)'
        
    elif 'NK' in markers_set or 'CD56+' in markers_set:
        return 'NK cells'
        
    elif 'gran' in markers_set or 'neutro' in markers_set:
        return 'Granulocytes'
        
    elif 'lymphocyte' in markers_set:
        return 'Lymphocytes (Unspecified)'
    elif 'WBC' in markers_set or 'CD45+' in markers_set:
        return 'Leukocytes (Unspecified)'
        
    return 'Unclassified / Potential Error'

df = pd.read_csv('cell_definitions_with_normalized.csv')
#Normalized_Definition was the population created after cell ontology cleaning and before grouping into broader categories
df['Intermediate_Population'] = df['Normalized_Definition'].apply(classify_intermediate)
df.to_csv('results_intermediate.csv', index=False)

# quantities = df['Intermediate_Population'].value_counts().reset_index()
# quantities.columns = ['Agrupamento', 'Quantidade']
# print(quantities.to_string(index=False))

##AFTER THIS CODE, I MANUALLY CHECKED EACH GROUPING TO SEE IF IT MADE SENSE AND THE FINAL GROUPING IS "GROUPED_POP.CSV"