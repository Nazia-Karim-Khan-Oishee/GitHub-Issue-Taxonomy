# Github Issue Taxonomy - Multi class text classification using pre trained BERT model.

## Installation

Before running the project, make sure you have Python installed. You can download Python from [here](https://www.python.org/downloads/).

Next, install the required libraries using the following command:
```bash
pip install torch pandas scikit-learn transformers tokenizers tqdm matplotlib seaborn
```
First download the pretrained model from [this link](https://drive.google.com/file/d/13JqaIYCr3pky4PF_OeEXj-YvqmKGs2nC/view?usp=drive_link) in the Code/notebook folder

To run the prediction using the pre-trained model, execute the following command:
```bash
python ImplementationOfModel.py
```
Input a text when prompted, and the model will provide the classification result.

## Implementation

To implement the model from scratch or explore the code, follow these steps:

1.Install Python and the required libraries mentioned in the Installation section.

2.Install Jupyter Notebook. You can install it using the following command:
```bash
pip install jupyter
```
3.Download BERT base uncased model from Hugging Face.You can download it from [here](https://huggingface.co/bert-base-uncased).

4. Download the datasets from kaggle and store them in the /Dataset folder. 

5.Navigate to the code folder and then the notebook folder. Open the model_training.ipynb file in Jupyter Notebook for the full implementation.

## Dataset

The dataset is collected from kaggle.You can download it from [this link](https://www.kaggle.com/datasets/anmolkumar/github-bugs-prediction/data?select=embold_train_extra.json)

## Model Assessment

All the model assessment data are provided in the repository as png files or in the notebook itself. We have gained almost 73% accuracy as of now. 



