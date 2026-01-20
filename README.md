**Unsupervised Discovery of Market Regimes**

This repository contains the research and implementation of unsupervised machine learning models to identify latent market regimes (Calm, Transitional, and Stressful) using historical financial data from 1995 to 2024.

The project compares Static Clustering (Gaussian Mixture Models) with Temporal Modeling (Hidden Markov Models) to evaluate which approach provides more stable and interpretable signals for financial risk management.



**📊 Key Findings**

Temporal Awareness Matters: The Hidden Markov Model (HMM) produced significantly more persistent regimes (~50-week average duration) compared to the Gaussian Mixture Model (~8-week average).

Economic Consistency: Both models successfully identified major historical crises (2008 GFC, 2020 COVID-19) as "Stress" regimes characterized by high VIX levels and negative equity returns.

Methodological Trade-Off: Static models (GMM) are more reactive to sudden market shocks but suffer from "flickering" signals, while HMMs provide a smoother, more actionable macro-regime signal.



**🛠️ Repository Structure**

├── data/

│   └── all CSV files

├── figures/

│   └── pictures here

├── notebooks/

│   ├── all the notebooks

├── paper/

│   └── Unsupervised_Market_Regimes.pdf     

└── README.md


**All of the main information about the research is in the paper**
