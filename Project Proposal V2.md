# Project Proposal: V2


## Goal

Learn discriminative shape representations from 3D vertebra meshes that support supervised prediction of species and spinal position, with hierarchical taxonomic evaluation.

This project will build upon the existing work done using DeepSDF and NSM to train a model for the classification of lizard vertebrae. Specifically, the model will be able to predict the top five species for a given specimen, the spinal position of the fossil, and any other relevant phylogenetic information. 

## Background

While 3D shape classification may be a young topic by relative standards, there exists a substantial body of research driving it forward and proving its effectiveness across many domains. Object detection and classification has particular use for navigation and autonomous driving, medicine, robotics, virtual and augmented reality, and more. Recent advances in 3D object classification have been driven in large part by deep learning architectures operating on a variety of input data types: voxel grids, multi-view projections, point clouds, meshes, or some hybrid there of. Each approach has its strengths, but ultimately all reside to modeling the input shape as a discrete sample rather continuous structures, limiting fidelity. 

Implicit generative models, such as DeepSDF and NSM, instead model geometry as continuous functions. The model then learns latent spaces that encode shape, and have been shown to offer strong results in reconstruction and interpolation tasks. This makes them especially suitable for biological contexts where specimen are often rough, noisy, or incomplete. Recent efforts to expand access to 3D data, such as oVert, has created unprecedented opportunities for impact in fields such as paleontology.

## Prior Work

When using generative implicit models like DeepSDF or NSM, it’s common to perform classification on the latent encodings using KNN with either cosine or Euclidean distance metrics. While straightforward, this approach has some limitations. They require making some assumptions about the latent space (linearly separable by distance metrics, semantically similar shapes cluster together) and do not make any special distinctions for hierarchical or multi-task objectives. Prior research has shown that performance often improves when a lightweight supervised classifier is trained on top of the representations rather than relying solely on distances.

As a proof of concept, I conducted an experiment to evaluate whether a learned classifier could outperform or complement a distance-based approach. I found that a supervised classifier can achieve strong performance (100% accuracy in the demo), and that results differed slightly between cosine and Euclidean distance measures, showcasing that this choice can have affects on downstream predictions. Going forward, experimentation is required with multiple types of classifiers (SVM, Random Forest, MLPs, Logistic Regression) as well as hierarchical learning methods.

## Planned Steps

### Establish and Analyze Baseline Architecture

Accomplishing the goals set out in this proposal will require a solid understanding of the current approach. 

To establish a strong baseline, I will first dissect the existing DeepSDF / NSM architecture to understand how latent shape representations are formed. DeepSDF models shapes as continuous signed distance functions using an 8-layer fully connected network in which a latent vector is concatenated with spatial coordinates and passed through multiple layers with skip connections. NSM adds onto this with a few encoder variants. The initial implementation specific to this project leverages a triplanar implicit decoder that transforms a 512-dimensional latent vector into feature maps on the XY, YZ, and XZ planes. To predict surface distances, points are sampled and projected onto the three planes, then interpolated to obtain local features, and passed to an implicit MLP. 

Building on this architectural understanding, I will systematically evaluate classification from the learned latent space. The current approach relies primarily on cosine or Euclidean distance for nearest-neighbor retrieval. I will compare these distance metrics against lightweight supervised classifiers trained directly on the latent codes to test whether learned decision boundaries improve species and spinal-region prediction. Additionally, I will experiment with metric learning objectives to reshape the latent space in an attempt to more cluster data by species and spinal position. Finally, I will explore multi-task and hierarchy-aware extensions to attempt to predict specials and spinal position together.

## Possible Extension Areas

- Hierarchy-aware loss. Can we somehow measure taxonomic distance into training/evaluation to inform when the model correctly identifies genus but not species (example)? Done correctly, this should reduce the severity of mistakes.
- Confidence scores.
- Can we provide some sort of interactive solution, either via a Jupyter Notebook or a custom UI, that allows users to try the model themselves?

## Metrics for Success

I will assess how additions/changes (metric learning, multi-task learning, etc.) affect metrics such as:

- Accuracy vs baseline in classification tasks
    - top-5 accuracy
    - spinal position prediction
    - taxonomy
- Classification Metrics
    - Instance Accuracy
    - Average Class Accuracy
    - Precision
    - Recall
    - F1
- Training time for new specimen
    - since NSM is an auto decoder, new examples have to be inference at evaluation time. Do any add-ons I make affect this step? Dramatically increasing compute time here would be negative
- Confusion Matrices:
    - Are there any insights or patterns from what the model gets wrong?
    - What sort of granularity can we have with this? Would like to have confusion matrices at multiple levels in the taxonomy

## Literature Review

### Provided Resources

- Review - Deep learning for 3D object recognition:
    - [https://doi.org/10.1016/j.neucom.2024.128436](https://doi.org/10.1016/j.neucom.2024.128436)
    - key takeaways:
        - Nuances and challenges of working with varying data types
        - Overviews of the field
        - Talks about advantages of rotation invariant strategies; worth exploring?
        - Outlines good classification metrics (Instance Accuracy,  Average Class Accuracy, Precision, Recall, F1)
        - Establishes some SOTA baselines on key data sets (ModelNet 40, 10, etc)
- Review - Neural 3D shape classification:
    - [https://doi.org/10.1109/TPAMI.2021.3102676](https://doi.org/10.1109/TPAMI.2021.3102676)
    - key takeaways:
        - code is made available in case we want to do our own baseline testing
        - establishes some SOTA (2022) baselines
        - call outs to data sets we can reference
            - ModelNet 40, 10
            - ShapeNet Core
        - overview of different approaches (point cloud, mesh, voxel, etc.)
- Integrating cartesian coordinates and feature vectors:
    - [https://doi.org/10.3390/rs15010061](https://doi.org/10.3390/rs15010061)
    - key takeaways:
        - overview of how implicit representation works
            - zero-level isosurface, etc.
        - justification for using implicit fields as discriminative geometry
            - not just for reconstruction; can improve classification
        - rotation invariance
        - feature engineering and distance field sampling
            - differences between local UDF and global SDF
- Blackburn et al. 2024, BioScience.
    - [https://doi.org/10.1093/biosci/biad120](https://doi.org/10.1093/biosci/biad120)
    - key takeaways:
        - OpenVertebrate (oVert) Thematic Collections Network
        - Impact on field
        - Where data comes from
        - Vertebrae encode meaningful phylogenetic information
        - Vertebral shape can distinguish clades even when external morphology is missing
        - More things to look into:
            - Confusion between adjacent vertebral positions
            - Whether latent space reflects axial gradients
- Gatti et al. 2025, IEEE TMI.
    - [https://doi.org/10.1109/tmi.2024.3485613](https://doi.org/10.1109/tmi.2024.3485613)
    - key takeaways:
        - Explains NSM as an extension of DeepSDF
        - Auto-decoder framework
        - Different representation of anatomical structure
        - Justification for our approach
- Park et al. 2019, CVPR.
    - [https://doi.org/10.48550/arXiv.1901.05103](https://doi.org/10.48550/arXiv.1901.05103)
    - key takeaways:
        - Defines DeepSDF; core to our implementation
        - Continuous implicit representation of 3D Shapes
        - Compact Latent embedding per shape
        - Auto-decoder
        - Latent space structure reflects shape similarity
- Porto et al. 2026, in prep.
    - [https://github.com/agporto/ATLAS](https://github.com/agporto/ATLAS)
    - Unclear how/when I personally will use this but for some sort of mesh analysis

### Self Discovery

- Urbani et al. 2024, “Harnessing Superclasses for Learning Hierarchical Databases”
    - [https://doi.org/10.48550/arXiv.2411.16438](https://doi.org/10.48550/arXiv.2411.16438)
    - key takeaways:
        - hierarchy-aware loss
            - should aid with my goals:
                - predict species
                - predict spinal position
                - respect phylogenetic hierarchy
                - reduce severity of taxonomic mistakes
- Garg et al. 2022, “Learning Hierarchy Aware Features for Reducing Mistake Severity”
    - [https://doi.org/10.48550/arXiv.2207.12646](https://doi.org/10.48550/arXiv.2207.12646)
    - Also may be helpful in handling taxonomy tree and/or phylogeny
    - Also addresses existing approaches to leveraging hierarchical taxonomy for classification tasks.
    - key takeaways:
        - Another loss function to experiment with
            - Jensen-Shannon Divergence paired with a geometric loss
        - Helps address mistake severity
        - Tune the latent space to contain more feature info, such as phylogenetic structure
- Metric Learning: [https://contrib.scikit-learn.org/metric-learn/introduction.html](https://contrib.scikit-learn.org/metric-learn/introduction.html)
    - key takeaways:
        - latent space fine tuning
        - varying mistake severity