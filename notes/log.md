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

**Focus:** Lexical analysis  
**Work done:**

- Fixed urls problem.
- Understood Spacy and wrote function for tokenization.
- Wrote function for plotting frequencies.

---

## 📅 2025-10-16

**Focus:** Lexical analysis  
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

**Focus:** Lexical analysis and characteristic term analysis  
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

**Focus:** Environment set up for model training  
**Work done:**

- Installed torch, sentencepiece, transformers.
- Installed c++ stuff for fairseq.
- Installed fairseq.

**Observations:**

- I don't think I'll need fairseq. I can use huggingface transformers to handle
  the model training

---

## 📅 2025-10-21

**Focus:** understanding training pipeline and Kaggle notebook  
**Work done:**

- Understood how to set training pipeline.
- Read basic workflow to train with huggingface transformers.
- Read Kaggle notebook guidelines.

---

## 📅 2025-10-22

**Focus:**  
**Work done:**

- Finished reading Kaggle notebook guidelines (Docker containers explanation).
- Considered Kaggle vs Colab.
- Studied how to use Huggingface Datasets and Transformers for the training
  pipeline.

**Observations:**

- I'll try with Kaggle first, if I have problems I'll switch to Google Colab
- I can use Huggingface Datasets and Transformers to split the dataset into
  train and test, tokenize and train.
