



## Automatic Grading of Student Code with Similarity Measurement

### Description

This paper presents SimGrader system, a code grading system that grades student code based on the
measurement of similarity. We extract the static features, runtime features and semantic features of the code to comprehensively grade the code.

### Environment

Python 3.7.6

Pytorch 1.7.0

### Requirements

1. Install the required dependencies 

   ```
   pip install -r requirements.txt
   ```

   

2. Install cpplint tool 

   ```
   pip install cpplint
   ```

   

3. Install cppcheck tool-- https://cppcheck.sourceforge.io/

### Usage

- ##### Enhancing Discrimination With Contrastive Learning

  train the representation encoder using the "Enhancing Discrimination" folder:

  1. Calculate tree edit distance for labeled 

     ```
     python CalculateTED.py
     ```

  2. Using contrastive learning to train code representation encoders

     ```
     python transformation/Transformation.py
     python encoder/pretrain.py
     ```

  3. Predicting code closeness to fine-tuning encoder

     ```
     python encoder/train.py
     ```

- ##### Feature Extraction

  extract the features using the "Feature " folder:

  ```
  python SimilarityFeature/CalSimilarity.py
  python StaticFeature/CalFeatures.py
  ```

- ##### Grading student code

  grade student code using the "Grade" folder:

  ```
  python Grade/GradingWithSim.py
  python Grade/GradingWithML.py
  ```

  

