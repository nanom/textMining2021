# VizWiz-VQA disaggregation: A new set of visual question classes.


## Overview ([See full report here!](full_report.pdf))
In this work, through the exploration of different clustering strategies and morpho-syntactic analysis, a new set of eight main categories is presented and proposed, which are used to fine-tune a automatic classification model and in this way to be able to reanalyze the original set from a new perspective. It should be noted that this research sets aside the visual modality 'V', to focus on the 'QA' part of VQA, with the aim of disaggregate the majority classes, to facilitate the understanding of the nature of the questions, and the reasons for why many of these questions cannot be answered.

## Directory content:

* [`full_report.pdf`](full_report.pdf): Full description of project.

* [`clustering_first_approach.ipynb`](clustering_first_approach.ipynb): This notebook, contains the coding and the development of the first approximations for the disaggregation of the questions in different clusters. The evaluation of the results is carried out visually. Subseccion 3.3 in full report.

* [`clustering_exploration_best_methods.ipynb`](clustering_exploration_best_methods.ipynb): In this notebook, it is culminated by selecting the best clustering strategy used that lays the foundations for the training of the classification model. In this, the results obtained are evaluated using semi-systematized methods. Subseccion 3.4 in full report.

* [`classification_model_training.ipynb`](classification_model_training.ipynb): After the proposal of the eight new classes that identify each of the final clusters obtained, Subsection 4.1 in full report, in this notebook four different combinations of classification models are trained and tested. Subsection 4.2 in full report. 

* [`testing_of_final_models.ipynb`](testing_of_final_models.ipynb): This notebook contains everything you need to test the different trained models and classify any type of question of interest. In it, there are several functions that automatically download the complete set of training and validation questions from the VizWiz-VQA dataset, in order to perform different queries and obtain statistical graphics of the obtained predictions. Unlike the results detailed in Section 5, here all models can be tested, even initial versions trained on a list of similar categories that are not detailed in the report. Feel free to try it out!