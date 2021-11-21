# Avoiding Bubbles

Hello!

All of the project dependencies are encapsulated in a conda environment named `avoiding_bubbles`. Whatever system is hosting this project will need an installation of anaconda or miniconda if you would like to install and activate the `avoiding_bubbles` environment. Setup instructions are below. 

## Setup

### Prerequisites

Conda installation instructions can be found here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### Creating conda env

From the project root run:

`conda env create -f environment.yml && conda activate avoiding_bubbles`

Next open the jupyter notebook:

`jupyter notebook`

You should now be able to browse to http://localhost:8888 and view the research project. All of the interesting things you will want to review live in the `avoiding_bubbles.ipynb` jupyter notebook. Navigate there within the jupyter UI and run all the cells!

## Project Structure
```text
avoiding_bubbles                      <- Project Root
├── data                              <- A place for storing CSVs accessed by the project notebook
|   └── *                 
├── utils                             <- A place for handy plotting and helper methods
|   └──  *
├── README.md                         
├── environment.yml                   <- An export of the conda env
├── avoiding_bubbles.ipynb            <- Jupyter notebook where all the project research lives
└── avoiding_bubbles_11-18-2021.pdf   <- A pdf export of the research notebook in case of Murphy's law
```

## Note on the `data/` Dir
All of the data generated by the notebook is cached in the data/ directory. If you would like the notebook to fetch/generate fresh data, you can delete the universe.csv and/or the confidence.csv file(s). The notebook should detect that the file(s) no longer exist and refetch or regenerate accordingly.
The Fama French Factor data will not regenerate, so don't delete those.

## FAQ
Q: Whoa whoa whoa, I don't want to do all this setup! 🙅 Can I just review a pdf export of your notebook?

A: Yes! There is an export of the latest version saved to a file named [`avoiding_bubbles_11-18-2021.pdf`](https://github.com/Boulder-Investment-Technologies/avoiding_bubbles/blob/main/avoiding_bubbles_11-18-2021.pdf). Alternatively, you can view the notebook via GitHub's notebook renderer here: [avoiding_bubbles.ipynb](https://github.com/Boulder-Investment-Technologies/avoiding_bubbles/blob/main/avoiding_bubbles.ipynb).
