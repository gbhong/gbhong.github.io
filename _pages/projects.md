---
layout: "single"
permalink: /experience/
title: "Research Projects"
excerpt: "experience.md"
author_profile: true
---

## User-specific information extraction based on BART summarization

#### *(in progress)*

{% include figure image_path="/assets/images/user_att_ext_1.png" %}

{% include figure image_path="/assets/images/user_att_ext_2.png" %}

- Research goal: To develop a model extracting user attributes from dialogue and persona.
  - Fine-tuned BART for summarization model to generate predicates and entities.
  - Generated a predicate first, because I thought that entities are bound to predicate.
  - Increased the prediction accuracy from 0.27 of the baseline model to 0.47.
- Insights
  - Extract user information easily from given dialogue text to pass as input for knowledge graph, as well as 
  - Lack of commonsense reasoning: The model could not extract that *‘I have father’* from the sentence *‘My father likes watching baseball’*, while successfully extracted the triplet *‘my father, has_hobby, watching baseball’.*

## Biomedical predicate classification using SemMedDB

{% include figure image_path="/assets/images/BioPREP_Overview.png" %}

- Research goal: To build relation extraction model between unseen biomedical entities.
  - Opened dataset, **BioPREP**, with 165,799 sentences, each labeled with 28 predicates.  
  - Fine-tuned BioBERT for predicate classification.
  - Analyzed the difference of model's performance depending on seven semantic groups defined by FrameNet.  
- Insights
  - Lack of deeper understanding: The model still struggled in capturing relations when the input sentence is quite long or contains complex local features occurring a subtle difference in meaning. For instance, our model extracts the relation name as *‘Occurs in’* from the input sentence *‘The occurrence of chromosome PHENOTYPE in PHENOTYPE may be associated with incomplete manifestation of the syndrome’*, while the ground truth for relation name is *‘Process of’*.  
    Though our pre-trained model showed the best performance, I think that our model does not fully understand given biomedical sentences semantically in some cases, only capturing the superficial correlation to infer target labels.

## WR / ERA / AVG Prediction model for Korean Baseball Teams

#### Grand Prize (Minister's Award) at Big Contest 2020, National Information Society Agency, Korea

{% include figure image_path="/assets/images/BigCon_overview.png" %}

- Project goal: To build a system to predict Win Rate, Batting Average, and Earned Run Average for Korean baseball teams.
  - **Two-stage architecture**: Predict ERA and AVG first with LSTM. 2) Infer WR using ‘Predicted ERA’ and ‘Predicted AVG’ as newly added features, with LightGBM.
  - Made embedding vectors by using scaled raw records and a multiple of derived features.
  - 26 consequential embedded vectors were used as the inputs of LSTM. Pooled the last hidden state of the LSTM layer to predict WR, ERA and AVG.
  - Scaled moving average of 26 consequential game features were used as input in LightGBM.
  - Best performance across 1,500 teams participated.

## Fake News Detection Challenge

### Winning a prize at NLP Competition, NH Investment & Securities, Korea

{% include figure image_path="/assets/images/fakenews_overview.png" %}

{% include figure image_path="/assets/images/fakenews_rnn.png" %}

- Project goal: To develop a model that filters fake news having no relation with main news contents.
  - Designed two-stage sequence classification algorithm with bidirectional RNN and ELECTRA.
  - Analyzed the relationship between news headline and each body sentences.
  - Improved time complexity by dynamic padding and uniform length batching.
  - **1st** on public leaderboard, 5th on private leaderboard across 1,000 teams participated.