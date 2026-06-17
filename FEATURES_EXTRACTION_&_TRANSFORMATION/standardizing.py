import pandas as pd
import re

def normalize_marker(marker):
    #Standardizing marker names based on cell ontology rules provided by immune space

    print(f"Original marker: '{marker}'")
    m = marker.strip()
    m_lower = m.lower()
    print(f"Marker after: '{m_lower}'")
    
    #Everything I found in the data that is synonym aparently
    synonyms = {
        'singlet': ['sing', 'singlets', 'singlet', 'doublet_excluded', 'sing-f', 'intact_singlet', 'single_cells', 'singlet_gate'],
        'lymphocyte': ['ly', 'lymp', 'lymph', 'lymphocyte', 'lymphs', 'lymphocytes', 'lymo', 'lymphos', 'lym'],
        'monocyte': ['mo', 'mono', 'monos', 'mnc', 'monocytes', 'pmo', 'monocyte'],
        'gran': ['gran', 'grans', 'granulocytes', 'granulocyte', 'granulo', 'pmns'], 
        'intact': ['intact_cells', 'intact_cells_population', 'intact'],
        'viable': ['live', 'annexin-', 'live/dead stain', 'live/dead', '7aad-', 'pi-', 'dapi-', 'viability', 'viable'],
        'proliferated': ['cfse-', 'tracerviolet', 'cfse_low', 'cfse_dim', 'ctv_low', 'celltrace_low', 'proliferating'],
        'neutro': ['neutro', 'neutrophil', 'neutrophils', 'neutros', 'pmn', 'neutr', 'neu'],
        'naive': ['naive', 'naïve', 'nv'],
        'EM': ['em', 'effector_memory', 'effector-memory'],
        'CM': ['cm', 'central_memory', 'central-memory'],
        'TEMRA': ['temra', 'emra'],
        'NK': ['nk', 'nk_cells', 'nkcells'],
        'NKT': ['nkt', 'nkt_cells'],
        'plasmablast': ['pb', 'plasmablast', 'plasmablasts', 'nplasmablast'],
        'plasma_cell': ['pc', 'plasma_cells', 'plasmacell'],
        'mDC': ['mdc'],
        'pDC': ['pdc'],
        'mo2': ['mo2'],
        'mo3': ['mo3'],
        'TH': ['th'],
        'TFH': ['tfh'],
        'PB': ['pb'],
        'NK': ['nk'],
        'WBC': ['wbc', 'swbc'],
        'TH': ['th'],
        'TH1': ['th1'],
        'TH2': ['th2'],
        'TH17': ['th17'],
        'TH1-17': ['th1-17', 'th117'],
        'TFH': ['tfh'],
        'TFH1': ['tfh1'],
        'TFH2': ['tfh2'],
        'TFH17': ['tfh17'],
        }
    
    m_base = re.sub(r'(_bright|_dim|_int|[+\-~]+)$', '', m_lower) #removing the state of the marker to check the synonym only with the base name (ex: CD3+ -> CD3)

    #for oficial_name, variations in synonyms.items():
    for standard, variations in synonyms.items():
        if m_base in variations or m_lower in variations:
            
            state = m_lower.replace(m_base, '') if m_base in variations else '' #recovering the state. Now we figure out the base, can remove from the original marker to get only the state
            if state == '':
                return standard
            else:
                return standard + state


    print(f"Marker after synonym check: '{m}'")
            
    #RUles to states -> I'm using binary rules to decrease the granularity, istead of having ++, +-...
    state_mappings = [
        (r'(neg)$', '-'),          
        (r'(pos)$', '+'),                
        (r'(dim|lo|di)$', '+'),                 
        (r'(int|medium|med)$', '+'),  
        (r'(bright|hi|bri|high|br)$', '+')          
    ]

    for pattern, replacement in state_mappings:
        if re.search(pattern, m):
            # print(re.sub(pattern, replacement, m, flags=re.IGNORECASE))
            #Subtitution keeping the original marker name
            m = re.sub(pattern, replacement, m, flags=re.IGNORECASE)
    
    if "+" not in m and "-" not in m and "+-" not in m and "+~" not in m and "++" not in m:
        m = m+"+"  #Rule based on immune space, if there's no state, we assume it's positive (ex: CD3 -> CD3+)

    #Some specific rules for markers that are commonly mistyped in the data and were breaking the code
    m = re.sub(r'^(dr|hladr)([+\-~]*)$', r'HLA-DR\2', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?1([+\-]*)$', r'CD1c\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?2([+\-]*)$', r'CD303\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?3([+\-]*)$', r'CD141\1', m, flags=re.IGNORECASE)
    m = re.sub(r'^bdca-?4([+\-]*)$', r'CD304\1', m, flags=re.IGNORECASE)
    
    print(f"Marker after state mapping: '{m}'")
    print("-" * 50)
    return m

def parse_definition(definition):
   
    #Cleaning and normalizing markers
    if pd.isna(definition):
        return ""

    #Removing extra text in the definitions
    def_clean = re.sub(r'\b\d+-(?=[a-zA-Z])', '', str(definition), flags=re.IGNORECASE)
    def_clean = re.sub(r'\bQ\d+:\s*', ' ', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'\bnQ\d+:\s*', ' ', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r',Freq\.? of.*', '', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'_?pP', '', str(def_clean), flags=re.IGNORECASE)  # Removing 'pP' that was not indentify as a marker
    
    #Avoiding remove "cells"
    def_clean = re.sub(r'\bplasma cells?\b', 'plasmacell', str(def_clean), flags=re.IGNORECASE)
    
    #Removing words that are not useful for the model and can be noise, but without removing the word "cell" or "lymphocyte" for example, since they are important to identify the population
    def_clean = re.sub(r'\b(AND|OR|Sum|of|cells?|small|count|SCC|B)\b', ' ', def_clean, flags=re.IGNORECASE)
    
    #SPecific rules for common mistakes in the data
    def_clean = re.sub(r'\bsmall\b', ' ', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'\bmo-([a-zA-Z0-9])', r'\1', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'\bNeu-CD', 'Neu, CD', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'HLA-DR', 'HLADR', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(f'K1-67', 'Ki67', def_clean, flags=re.IGNORECASE)

    def_clean = re.sub(r'sing-F', 'singlet', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'2-WBC', 'WBC', def_clean, flags=re.IGNORECASE)    
    def_clean = re.sub(r'lymono', 'lymphocyte, monos', def_clean, flags=re.IGNORECASE)

    def_clean = re.sub(r'\bdn\b', 'CD14-, CD16-', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo1', 'CD14+, CD16-', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo2', 'CD14+, CD16+', str(def_clean), flags=re.IGNORECASE)
    def_clean = re.sub(r'mo3', 'CD14+, CD16+', str(def_clean), flags=re.IGNORECASE)

    def_clean = re.sub(r'(TH|TFH)/([0-9\-]+)', r'\1\2', def_clean, flags=re.IGNORECASE)
    def_clean = re.sub(r'[^a-zA-Z0-9\s+-]', ' ', def_clean)  # Removing special characters except for +, -, and spaces
    def_clean = re.sub(r'(pos|neg|dim|lo|di|bright|hi|br|bri)(?=[A-Z])', r'\1 ', def_clean)
    def_clean = re.sub(r'([+-])(?=[a-zA-Z])', r'\1 ', def_clean)
    #SPliting definition into markers
    markers = re.split(r'\s+', def_clean.strip())
    
    normalized_markers = [normalize_marker(m) for m in markers if m.strip()]
    marker_set = set(normalized_markers)

    #More rules that I found to decrease the granularity creating generic features
    if 'CD45RA+' in marker_set and 'CCR7+' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA+', 'CCR7+']]
        normalized_markers.append('naive')
        print(f"Found naive signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")
        
    elif 'CD45RA-' in marker_set and 'CCR7-' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA-', 'CCR7-']]
        normalized_markers.append('EM')
        print(f"Found EM signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")    
        
    elif 'CD45RA-' in marker_set and 'CCR7+' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA-', 'CCR7+']]
        normalized_markers.append('CM')
        print(f"Found CM signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")
        
    elif 'CD45RA+' in marker_set and 'CCR7-' in marker_set:
        normalized_markers = [m for m in normalized_markers if m not in ['CD45RA+', 'CCR7-']]
        normalized_markers.append('TEMRA')
        print(f"Found TEMRA signature in definition: '{definition}' -> Normalized: '{normalized_markers}'")

    if 'CD19+' in marker_set or 'CD20+' in marker_set:
        if 'IgD+' in marker_set and 'CD27-' in marker_set:
            normalized_markers.append('B_naive')
        elif 'IgD-' in marker_set and 'CD27+' in marker_set:
            normalized_markers.append('B_switched_memory')
        elif 'IgD+' in marker_set and 'CD27+' in marker_set:
            normalized_markers.append('B_unswitched_memory')

    if 'HLADR+' in marker_set and 'CD11c+' in marker_set and 'CD14-' in marker_set:
        if 'CD1c+' in marker_set: normalized_markers.append('mDC1')
        if 'CD141+' in marker_set: normalized_markers.append('mDC2')
    if 'HLADR+' in marker_set and 'CD123+' in marker_set and 'CD11c-' in marker_set:
        normalized_markers.append('pDC')

    if 'CD303+' in marker_set or 'CD304+' in marker_set or \
       ('CD123+' in marker_set and 'CD11c-' in marker_set):
        normalized_markers.append('pDC')

    if 'pDC' not in normalized_markers:
        
        #Subtype mDC1 
        if 'CD1c+' in marker_set:
            normalized_markers.append('mDC1')
            
        #Subtype mDC2 
        elif 'CD141+' in marker_set:
            normalized_markers.append('mDC2')
            
        #generic mDC
        elif 'CD11c+' in marker_set and 'CD123-' in marker_set:
            normalized_markers.append('mDC')
    
    #Removing duplicates
    normalized_markers = list(set(normalized_markers))
    normalized_markers.sort(key=str.lower)
    
    return ", ".join(normalized_markers)


df = pd.read_csv("population_definition_and_unit_counts.csv")

column_name = 'Population Definition Reported' #The column that we are using to extract the features, can be changed if we want to apply the same process to another column
df['Normalized_Definition'] = df[column_name].apply(parse_definition)
df.to_csv("cell_definitions_with_normalization.csv", index=False)

# unique_results = df['Normalized_Definition'].dropna().unique()

# print(f"\n{len(unique_results)} UNIQUE RESULTS AFTER CLEANING AND NORMALIZATION:\n")

# df_uniques = pd.DataFrame(unique_results, columns=['Normalized_Definition'])
# df_uniques.to_csv("unique_results.csv", index=False)

