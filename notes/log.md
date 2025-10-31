# Project Journal – MultiPRIDE (Italian)

## 📅 2025-10-13

**Focus:** Project setup and repo reorganization

**Work done:**

- Refactored repo structure.
- Checked initial dataset overview.

**Observations:**

- 207 reclamatory tweets out of 1086 (≈19%).
- Average tweet length = 34 tokens.
- Average bio lenght = 13 tokens.

**Next steps:**

- Begin lexical/semantic exploration.
- Identify frequent terms and emojis.

---

## 📅 2025-10-14

**Focus:** Lexical analysis.

**Work done:**

- Fixed urls problem.
- Understood Spacy and wrote function for tokenization.
- Wrote function for plotting frequencies.

---

## 📅 2025-10-16

**Focus:** Lexical analysis.

**Work done:**

- Refactored some code into functions.
- Considered frequency plots.

**Observations:**

- In the most frequent words of the non-reclamation tweets, there are no slurs.
  The words _gay_ and _trans_ can sometimes be used as slurs, but they usually
  have a neutral meaning.
- In the reclamation tweets, different variations of _frocio_ (e.g., frocio,
  forcio, forci, froci, frocia) are among the most frequent words. Some of these
  are misspelled, which could pose challenges during the model's training phase.
- Words that suggest reclamation, such as _pride_, are not particularly
  informative in distinguishing the label of the tweet.

---

## 📅 2025-10-19

**Focus:** Lexical analysis and characteristic term analysis.

**Work done:**

- Created a function to consider frequency analysis for hashtags and emojis.
- Modified plot_common function to plot relative frequencies of words, hashtags
  and emojis for better comparability.
- Implemented quick and dirty solution for characteristic term analysis.
- Plotted slurs frequencies.

**Observations:**

- Some emojis seem to be informative.
- Frocia and finocchio seem to be informative words.

---

## 📅 2025-10-20

**Focus:** Environment set up for model training.

**Work done:**

- Installed torch, sentencepiece, transformers.
- Installed c++ stuff for fairseq.
- Installed fairseq.

**Observations:**

- I don't think I'll need fairseq. I can use huggingface transformers to handle
  the model training.

---

## 📅 2025-10-21

**Focus:** understanding training pipeline and Kaggle notebook.

**Work done:**

- Understood how to set training pipeline.
- Read basic workflow to train with huggingface transformers.
- Read Kaggle notebook guidelines.

---

## 📅 2025-10-22

**Focus:** Platform exploration and Huggingface studying.

**Work done:**

- Finished reading Kaggle notebook guidelines (Docker containers explanation).
- Considered Kaggle vs Colab.
- Studied how to use Huggingface Datasets and Transformers for the training
  pipeline.

**Observations:**

- I'll try with Kaggle first, if I have problems I'll switch to Google Colab.
- I can use Huggingface Datasets and Transformers to split the dataset into
  train and test, tokenize and train.

---

## 📅 2025-10-24

**Focus:** Training notebook and src directory.

**Work done:**

- Started to write Training Notebook.
- Created files in src directory.

**Observations:**

- The stratified attribute of train_test_split works only if the selected column
  is of type ClassLabel. Now I'm not using stratified, but I should do it in the
  future to avoid imbalanced label between train and val sets.

---

## 📅 2025-10-25

**Focus:** Preprocessing functions.

**Work done:**

- Wrote tokenizer loader and tokenization functions.
- Applied the functions in the training notebook.

**Observations:**

- I changed the label column to LabelClass, thus I can use stratified in
  test_train_split.
- I have to understand how big the functions in the preprocessing file should
  be. I can refactor later anyway.

---

## 📅 2025-10-27

**Focus:** Study datasets, transformers and evaluation.

**Work done:**

- Studied all the pipeline on hugginface.
- Fixed the problem with src imports.

**Observations:**

---

## 📅 2025-10-28

**Focus:** Evaluation, Trainer and pipeline test.

**Work done:**

- Wrote evaluation function.
- Applied evaluation function to training notebook.
- Applied dinamic padding.
- Tested the whole pipeline in the training notebook.

**Observations:**

- The training seems to work, but there is a problem with the evaluation
  function. I need to better study how evaluation works.

---

## 📅 2025-10-29

**Focus:** Testing and Kaggle.

**Work done:**

- Tested the pipeline with small datasets.
- Wrote all the code on Kaggle.
- Studied how to interpret results.

**Observations:**

- The evaluation bug was probably related to the size of the datasets. I made
  some checks and everything seems to work fine. I'll test on Kaggle with a
  bigger dataset to be sure.
- To check learning and metrics curves I could use wandb. I need to read some
  documentation to understand how to use it.

**Next steps:**

- Upload training data on Kaggle.
- Test the pipeline on Kaggle.

---

## 📅 2025-10-31

**Focus:** Kaggle data mounting and test.

**Work done:**

- Uploaded the code on Kaggle.
- Tested on Kaggle, problems with libraries compatibility.
- Switched to Colab, tested the pipeline. It works fine.
- Set wandb on Colab for metrics visualization.

**Observations:**

- The run on Colab worked fine. The metrics showed that there is a problem of
  overfitting, probably due to the small training dataset. I need to get some
  visualization of the metrics and then try to understand how to fight
  overfitting.

**Next steps:**

- Change Hyperparameters to avoid overfitting.
- Test pipeline on Colab with metrics visualization (wandb).

---
