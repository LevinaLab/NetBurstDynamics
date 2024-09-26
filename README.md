# NetBurstDynamics
Fitting a simplified population bursting model to neural data. Code for Vinogradov et al. 2024. 

The repository to reroduce the analysis and visualizations from

"Effective excitability captures network dynamics across development and phenotypes"
Vinogradov et al., 2024


## Installation

Clone the repository
```bash
pip clone https://github.com/LevinaLab/NetBurstDynamics.git
```

Making a new conda environment and installing the dependencies with conda

```bash
env create --name NetBurstDynamics --file dependencies.yml
```

install the code in you local environment as 
```python
pip install -e . 
```

## Workflow
- Exploring the model dynamics
- Data processing and burst detection
- Model fitting 
- Visualization 


The project is being update to ensure compatibility.

## Project structure
- data/
- src/
- trained/
- scripts/
    - Figures/ 
    - DataProcessing/

Data folder should be populated from figshare directory
and 
data contains spikes from 24-well MEA recorded at DIV
[Figshare](https://figshare.com/projects/NetBurstDynamics/221734)


## Citation

```latex
@article{vinogradov2024effective,
  title={Effective excitability captures network dynamics across development and phenotypes},
  author={Vinogradov, Oleg and Giannakakis, Emmanouil and Buendia, Victor and Uysal, Betuel and Ron, Shlomo and Weinreb, Eyal and Schwarz, Niklas and Lerche, Holger and Moses, Elisha and Levina, Anna},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```


## Liscence for the code 
[Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)