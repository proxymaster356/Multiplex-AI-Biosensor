import pandas as pd
import numpy as np

def generate_biosensor_data(n=200):
    # Mapping Channels to Project Biomarkers
    # Ch1-4: AMR (blaNDM, mecA, vanA, KPC)
    # Ch5-8: Biofilm (icaADBC, AHLs, bap, c-di-GMP)
    # Ch9-12: Oncology (FadA, CagA, pks, miRNA-RNA)
    
    data = []
    for i in range(1, n + 1):
        sample = {'Sample_ID': f'S{i:03}'}
        category = np.random.choice(['Healthy', 'AMR', 'Biofilm', 'Oncology', 'Complex'], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        for ch in range(1, 13):
            # Determine if target is present
            is_present = False
            if (category == 'AMR' and 1 <= ch <= 4) or \
               (category == 'Biofilm' and 5 <= ch <= 8) or \
               (category == 'Oncology' and 9 <= ch <= 12) or \
               (category == 'Complex' and np.random.rand() > 0.4):
                is_present = True
            
            # Physics: Signal-OFF Logarithmic Decay [cite: 1848, 2606]
            # Baseline = 10nA, Max drop to ~0.5nA
            noise = np.random.normal(0, 0.1)
            if is_present:
                conc = np.random.uniform(10, 500) # Concentration in pM
                current = 10.0 - (2.1 * np.log10(conc)) + noise
            else:
                current = 10.0 + noise
            
            sample[f'Ch{ch}_nA'] = round(max(0.3, current), 3)
            
        sample['Diagnostic_Outcome'] = category
        data.append(sample)
    
    return pd.DataFrame(data)

# Generate and Save
df_200 = generate_biosensor_data(200)
df_200.to_csv('biosensor_200_samples.csv', index=False)
print("Dataset of 200 samples generated as 'biosensor_200_samples.csv'")