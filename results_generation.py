# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Create Table of Entity Overview

# Create a dictionary with the column data
data = {
    'Name (Entity)*': ['ML.ADECT.RELN_BFSN_Vendor',
                       '  HUB.PAPO.RELN_Vendor',
                       '  HUB.PAPO.BFSN_Outletcity_Metzingen_GmbH_Vendor',
                       'ML.ADECT.BFSN_PAYMENT',
                       '  HUB.PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal'
                       '  HUB.PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Head',
                       '  HUB.PAPO.BFSN_Outletcity_Metzingen_GmbH_Payment_Proposal_Line',
                       'ML.ADECT.RELN_PAYMENT',
                       '  HUB.PAPO.RELN_RE_Payment_Transaction',
                       '  HUB.PAPO.RELN_RE_Payment_Entry',
                       '  HUB.PAPO.RELN_RE_Payment_Line',
                       '  HUB.PAPO.RELN_Vendor_Ledger_Entry',
                       'ML.ADECT.RELN_PAYMENT_PROPOSAL',
                       '  HUB.PAPO.RELN_RE_Payment_Proposal',
                       '  HUB.PAPO.RELN_RE_Payment_Proposal_Entry',
                       '  HUB.PAPO.RELN_RE_Payment_Proposal_Line',
                       'ML.ADECT.TOTAL_PAYMENTS',
                       ],
    'Description': ['Summary vendor details (RELN and BFSN)',
                    'Vendor details (RELN)',
                    'Vendor details (BFSN)',
                    'Summary payment transactions (BFSN, incl. proposals)'
                    'Transaction general overview (BFSN)',
                    'Transaction master data (BFSN)',
                    'Transaction details (BFSN)',
                    'Summary payment transactions (RELN)',
                    'Transaction general overview (RELN)',
                    'Transaction master data (RELN)',
                    'Transaction details (RELN)',
                    'Additional vendor transaction specific details (RELN)',
                    'Summary payment proposal transaction (RELN)',
                    'Transaction general overview (RELN proposals)',
                    'Transaction master data (RELN proposals)',
                    'Transaction details (RELN proposals)',
                    'Total Summary all payment transactions',
                    ],
    'n**': [7513,
            6627,
            1643,
            40843,
            6347,
            241517,
            259643,
            77767,
            10490,
            47154,
            151239,
            317834,
            274,
            17960,
            52,
            331,
            118884],
    '# of features': [8,
                      14,
                      15,
                      45,
                      34,
                      48,
                      56,
                      45,
                      24,
                      27,
                      39,
                      16,
                      45,
                      20,
                      20,
                      53,
                      45]
}


# Create a DataFrame using the dictionary
df = pd.DataFrame(data)

# Add a footnotes to the DataFrame
footnote1 = '\\footnote{* Tables that are indented are subtables of the table above}'
footnote2 = '\\footnote{** n represents the number of observations per table}'

# Convert the DataFrame to a LaTeX table with the footnote included
latex_table = df.to_latex(index=False,
                          caption='List of entities, where transaction data is originated',
                          column_format='l') + footnote1 + footnote2

# Save the LaTeX table to a file
with open('Origin of Data, list of entities.tex', 'w') as file:
    file.write(latex_table)

print("LaTeX table has been saved to 'Origin of Data, list of entities.tex'.")

# %% Create Latex Tabel with the Overall Quality Score of each Syntheziser Model

# Create a dictionary with the column data
data2 = {
    'Model': ['GaussianCopula_Synthesizer',
              'CTGAN_Synthesizer',
              'TVAE_Synthesizer',
              'CopulaGAN_Synthesizer'
              ],
    'Overall Quality Score': ['71.56%',
                              '85.46%',
                              '87.26%',
                              '85.53'
                              ],
    'Column Shapes': ['82.25%',
                      '89.00%',
                      '91.10%',
                      '90.45%'             
                      ],
    'Column Pair Trends': ['60.87%',
                           '81.93%',
                           '83.42%',
                           '80.61%'
                           ]}

# Create a DataFrame using the dictionary
df2 = pd.DataFrame(data2)

# Convert the DataFrame to a LaTeX table
latex_table2 = df2.to_latex(index=False,
                            caption='Overall Quality Score Summary of each Syntheziser Model',
                            column_format='lrrr')

latex_table2


                                             

# %%
