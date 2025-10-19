# Project Journal – MultiPRIDE (Italian)

## 📅 2025-10-13

**Focus:** Project setup and repo reorganization  
**Work done:**

- Refactored repo structure
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

- fixed urls problem
- understood Spacy and wrote function for tokenization
- wrote function for plotting frequencies

## 📅 2025-10-16

**Focus:** Lexical analysis  
**Work done:**

- refactored some code into functions
- considered frequency plots

**Observations:**

- In the most frequent words of the non-reclamation tweets, there are no slurs.
  The words _gay_ and _trans_ can sometimes be used as slurs, but they usually
  have a neutral meaning.
- In the reclamation tweets, different variations of _frocio_ (e.g., frocio,
  forcio, forci, froci, frocia) are among the most frequent words. Some of these
  are misspelled, which could pose challenges during the model's training phase.
- Words that suggest reclamation, such as _pride_, are not particularly
  informative in distinguishing the label of the tweet.

## 📅 2025-10-19

**Focus:** Lexical analysis and characteristic term analysis  
**Work done:**

- created a function to consider frequency analysis for hashtags and emojis
- modified plot_common function to plot relative frequencies of words, hashtags
  and emojis for better comparability

**Observations:**
