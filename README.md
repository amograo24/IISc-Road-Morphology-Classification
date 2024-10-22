Hello 👋, The work in this repository is a part of my Research Internship at the **IISc Sustainable Transportation (IST) Lab** at the Indian Institute of Science, Bangalore. Since all work done in this internship is part of a publication that's in process, I am not allowed to share all components of my work.

This repo contains a much condensed version of the **Multi-Label Urban Road Morphology Classification** project. You can refer to the ```master-script.py``` that contains 
- **Dataset Class**
- **Augmentation/Pre-Processing Functions**
- **Epoch Runner**
- **Logging Mechanism (local and WandB)**

⭐️
I conducted a comparitive analysis on Residual Nets, EfficientNets, and DenseNets for this context. 
Our best model achieved a Hamming Score of 0.84 and and **F-Score of 0.866**.

🛰️
The LULC-Clustering-Archived folder contains some test scripts that apply Principal Component Analysis (PCA) as a pre-processing method on 12 spectral band images. Then, KMeans Clustering has been performed to identify classes for distinction between various land covers.

Happy to discuss more in-person!

