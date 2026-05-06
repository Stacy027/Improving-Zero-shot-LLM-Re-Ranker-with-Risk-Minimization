# Improving-Zero-shot-LLM-Re-Ranker-with-Risk-Minimization
Code for UR3 Method of "Improving Zero-shot LLM Re-Ranker with Risk Minimization" Paper

## Overview
In the Retrieval-Augmented Generation (RAG) system, advanced Large Language Models (LLMs) have emerged as effective Query Likelihood Models (QLMs) in an unsupervised way, which re-rank documents based on the probability of generating the query given the content of a document. However, directly prompting LLMs to approximate QLMs inherently is biased, where the estimated distribution might diverge from the actual document-specific distribution. In this study, we introduce a novel framework, UR3
, which leverages Bayesian decision theory to both quantify and mitigate this estimation bias. Specifically, UR3 reformulates the problem as maximizing the probability of document generation, thereby harmonizing the optimization of query and document generation probabilities under a unified risk minimization objective. Our empirical results indicate that UR3 significantly enhances re-ranking, particularly in improving the Top-1 accuracy. It benefits the QA tasks by achieving higher accuracy with fewer input documents.


<p align="center">
  <img width="358" height="362" alt="image" src="https://github.com/user-attachments/assets/99dbccd2-5f9f-4c1e-9e28-4bd59c86d4a6" />
