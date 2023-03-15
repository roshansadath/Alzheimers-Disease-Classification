# COMP6341 : Computer Vision

Alzheimer's disease is a neurodegenerative condition that causes memory impairment, especially in older individuals, and currently has no cure. 
Due to a large number of patients, it is not feasible to manually diagnose the disease in a timely and efficient manner. 
Although various diagnostic procedures are available, mistakes during the diagnosis process are common due to time constraints and the complexity of the disease. 
Thus, there is a pressing need for a precise and timely diagnosis of Alzheimer's disease. Our proposed project compares various deep learning-based medical imaging methods for diagnosing and classifying Alzheimer’s disease at different stages.

## Setting up your environment
1. Install the required packages with `pipenv install`
2. Download the data and add it to a folder named `data` at the root of the project

## Using this project
You can train a model using the `train` script

```shell
python -m alzheimerdetection.train alexnet
```

The progress of the training will be shown in your console