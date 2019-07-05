# MTL-face-analysis

This repository contains sample code of my master thesis which dealt with 
a deep multi task learning approach for various tasks in facial analysis.
Please note that this repository doesn't contain the used datasets an as such isn't 
intended to be a complete project.
Its purpose is to serve as an overview of how to implement the method i used 
(with some pseudo code).

A big problem while considering MTL is the question of how to combine the different 
tasks during training.
Usually a heuristic is used (eg. importance of tasks) or a grid search has to be done, 
which have numerous disadvantages.

As such we used the intrinsic and learnable homoscedastic uncertainty of the model 
given its specific tasks.



---------------------------------------------------------------------------------------------------------

### Influential Work

*   Learning homoscedastic uncertainty to weigh losses in the total loss function 
[Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) .

*   The model architecture was inspired by [Model Architecture](https://arxiv.org/abs/1603.01249).
    It introduces Hyperfeatures.