---
layout: default
title: Proposal
---

## Summary of the Project

The goal of this project is to develop an artificial intelligence system that assists in the detection and segmentation of brain metastases from multi-modal MRI scans. Brain metastases are a common complication of many primary cancers and are frequently treated using surgery or stereotactic radiosurgery, both of which rely on accurate identification and localization of tumors. Manual segmentation of brain metastases by radiologists is accurate but time-consuming and cognitively demanding, especially when patients have multiple small lesions distributed across the brain.

This project will use the Stanford Brain Metastases MRI dataset, which consists of 156 whole-brain MRI studies with four co-registered imaging modalities and expert-provided segmentation masks for a subset of cases. The input to the system will be 3D MRI volumes from multiple sequences (e.g., T1 pre-contrast, T1 post-contrast, T2 FLAIR), and the output will be a binary segmentation mask indicating the locations of metastatic tumors.

While brain tumor segmentation is often treated as a purely supervised learning problem, this project explores the use of reinforcement learning to guide aspects of the detection or refinement process. Specifically, reinforcement learning may be used to model an agent that sequentially explores the brain volume, focuses attention on candidate regions, or iteratively refines an initial segmentation produced by a supervised model. The overarching goal is to investigate whether reinforcement learning can improve detection accuracy, particularly for small or difficult-to-detect lesions, compared to standard baseline approaches.

## Project Goals

- **Minimum goal:**  
  Implement and train a baseline supervised learning model for brain metastasis segmentation using the labeled MRI scans. The model should successfully produce segmentation masks that overlap meaningfully with ground-truth annotations and establish a reference level of performance.

- **Realistic goal:**  
  Design and implement a reinforcement learning framework that operates on top of, or alongside, the baseline model. The RL agent will learn a policy for tasks such as navigating through 3D MRI volumes, selecting regions of interest, or refining segmentation boundaries. The resulting system should demonstrate measurable improvement over the baseline in at least one quantitative metric, such as Dice score or lesion-level recall.

- **Moonshot goal:**  
  Develop a reinforcement learning agent that learns an efficient and interpretable strategy for detecting small or sparse metastases across the entire brain volume. Ideally, the agent would demonstrate improved sensitivity to small lesions and reduced false negatives, while also providing insight into how sequential decision-making can complement traditional segmentation methods in medical imaging.

## AI/ML Algorithms

The project will use deep convolutional neural networks for supervised image segmentation, potentially inspired by encoder–decoder architectures commonly used in medical imaging. Reinforcement learning will be applied using a model-free approach with a neural function approximator, such as deep Q-learning or policy gradient methods, where the agent learns a policy for region selection, navigation, or segmentation refinement based on reward signals derived from overlap with ground-truth annotations.

## Evaluation Plan

Quantitative evaluation will be conducted using the labeled portion of the Stanford Brain Metastases dataset. Standard medical image segmentation metrics will be used, including Dice coefficient, precision, recall, and lesion-level detection accuracy. The primary baseline will be a supervised segmentation model trained without any reinforcement learning component. Additional naïve baselines may include simple intensity-based thresholding or single-modality models.

Experiments will compare baseline performance against the proposed RL-augmented approach under identical training and evaluation conditions. Cross-validation or held-out validation splits will be used to ensure fair comparison. We estimate that the reinforcement learning component could lead to modest but meaningful improvements in performance (e.g., 5–10% relative improvement in Dice score or improved recall for small lesions), particularly in challenging cases with multiple or small metastases.

Qualitative evaluation will involve visual inspection of segmentation results by overlaying predicted masks on MRI slices. These visualizations will be used to assess anatomical plausibility, boundary quality, and common failure modes. Additional qualitative analysis will include debugging and sanity checks, such as examining agent trajectories, visualizing which regions of the brain the RL agent focuses on over time, and evaluating behavior on simplified or toy examples. Successful results are expected to show more consistent localization of metastases and clearer refinement of tumor boundaries compared to the baseline model.

## AI Tool Usage

AI tools may be used during the project to assist with brainstorming model designs, understanding reinforcement learning concepts, debugging code, and improving documentation clarity. All dataset handling, model implementation, training, experimental design, and evaluation will be carried out by the project authors. AI-generated suggestions will be critically evaluated and adapted as needed, and no AI tool will be used as a replacement for independent implementation or analysis.

