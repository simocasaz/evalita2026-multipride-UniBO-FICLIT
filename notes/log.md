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

- Wrote tokenizer loader and tokenization functions
- applied the functions in the training notebook

**Observations:**

- I changed the label column to LabelClass, thus I can use stratified in
  test_train_split
- I have to understand how big the functions in the preprocessing file should
  be. I can refactor later anyway.

---

## 📅 2025-10-26

**Focus:**  
**Work done:**

**Observations:**

---

## 📅 2025-10-27

**Focus:**  
**Work done:**

**Observations:**

---

## 📅 2025-10-28

**Focus:**  
**Work done:**

**Observations:**

---

## 📅 2025-10-29

**Focus:**  
**Work done:**

**Observations:**
