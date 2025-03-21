#!/bin/bash

echo "Starting Data Preprocessing..."
python src/data_preprocessing.py 

echo "Starting Model Training..." 
python src/model_training.py 

# echo "Starting Model Explainability..."
# python src/model_explainability.py

echo "Pipeline completed! You can now run Streamlit, remember to push the code to github!" 