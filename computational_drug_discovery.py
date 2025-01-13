############################
# By: Ishraq Akbar
# Date: 2022/09/02
#-------------------
# This script uses data for SARS coronavirus to understand more about its biological activity, and develop a machine learning approach to interpret and understand how to develop a drug using
# Lipinski indicators against the virus
############################


# install chembl web service package
# ! pip install chembl_webresource_client

# Import libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import seaborn as sns
sns.set(style = 'ticks')
import matplotlib.pyplot as plt
from chembl_webresource_client.new_client import new_client


# Search for coronavirus
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
targets

# Retrieve bioactivity data for SARS coronavirus 3C-like proteinase 
selected_target = targets.target_chembl_id[4]
selected_target

# Retrieve activity reported as IC50 values for the coronavirus
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df = pd.DataFrame.from_dict(res)
df.standard_type.unique()
df.to_csv('bioactivity_data.csv', index=F)

# If any compounds have missing values from standard_value columns then drops them:
df2 = df[df.standard_value.notna()]
df2

# Process bioactivity data and label compounds as active, inactive, or intermediate based on IC50 values:
bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) <= 10000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")

# iterate molecule to list:
mol_cid = []
for i in df2.molecule_chembl_id:
    mol_cid.append(i)

# Iterate canonical_smiles to list:
canonical_smiles = []
for i in df2.canonical_smiles:
    canonical_smiles.append(i)


# iterate standard_value to list:
standard_value = []
for i in df2.standard_value:
    standard_value.append(i)

# Combine lists into dataframe:
data_tuples = list(zip(mol_cid, canonical_smiles, bioactivity_class, standard_value))
df3 = pd.DataFrame(data_tuples, columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])
df3 # View

# Export:
df3.to_csv('bioactivity_preprocessed_data.csv', index=F)



# Function to calculate Lipinski descriptors (inspired by https://codeocean.com/explore/capsules?query=tag:data-curation)
# The rule describes molecular properties important for a drug's pharmacokinetics in the human body (https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five)

def lipinski(smiles, verbose=F):
    moldata = []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arrange(1,1)
    i=0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Descriptors.NumHDonors(mol)
        desc_NumHAcceptors = Descriptors.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])
        
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
            i=i+1
        
        columnNames=["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        descriptors = pd.DataFrame(data=baseData, columns=columnNames)

        return descriptors

df_lipinski = lipinski(df.canonical_smiles)

# Combine dataframes:
df_combined = pd.concat([df, df_lipinski], axis=1)
df_combined # View

# Function to convert standard (IC50) value to pIC50 scale to account for uneven distribution of data points:
def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Convets nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', 1)

    return x

# Function to convert values > 100^6 will be fixed at 100^6, otherwise negative log values will become negative 
def norm_value(input_data):
    norm = []

    for i in input_data['standard_value']:
        if i > 100**6:
            i = 100**6
        norm.append(i)

    input_data['standard_value_norm'] = norm
    x = input_data.drop('standard_value', axis=1)

    return x

# Apply the normalization function, then pIC50 function:
df_norm = norm_value(df_combined)
df_final = pIC50(df_norm)
df_final

# Remove intermediate bioactivity class entries:
df_2class = df_final[df_final.bioactivity_class != 'intermediate']
df_2class # View

# Perform frequency plot of the two bioactivity classes:
plt.figure(figsize= 5.5, 5.5)
sns.countplot(x='bioactivity_class', data=df_2class, edgecolor = 'black')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('freq_plot_bioactivity.pdf') # Save file, comment out if not using


# Box plot comparing pIC50 between two bioactivity classes:
plt.figure(figsize = 5.5, 5.5)
sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.savefig('scatter_plot_bioactivity.pdf') # Save file, comment out if not using






