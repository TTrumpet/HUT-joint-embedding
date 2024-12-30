# Music Recommendation Using Exemplars and Contrastive Learning
Honors Undergraduate Thesis Project - Contrastive Learning

## News
Published!!! UCF STARS October 2024

## Setup
After cloning the repository, download ast-finetuned-audioset weights and bert-base uncased weights

Run download_audioset.py and download_musiccaps.py to obtain datasets necessary.

Run create_testing_set.py to create test dataset.

Run create_embedding_space.py with desired dataset to create embedding space for the model to learn from.

Run train_mulan.py to train the model.