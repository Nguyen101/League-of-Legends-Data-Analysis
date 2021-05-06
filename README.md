# League-of-Legends-Data-Analysis
# How to run:  
1. Follow this Jupyter Notebook to set up your environment : https://github.com/GonzagaCPSC322/U0-Introduction/blob/master/B%20Environment%20Setup.ipynb
2. Install tabulate: pip install tabulate  
3. Clone this repo  
4. Create a Heroku app or you can learn how to make a Heroku app by following this YouTube link https://www.youtube.com/watch?v=4eQqcfQIWXw by Dr. Sprint
6. Use this link to predict : http://league-of-legends-cpsc322-s1.herokuapp.com/predict?first_blood=1&first_tower=1&first_inhib=2&first_baron=1&first_dragon=1&first_riftherald=1
# Organization:
- Input dataset: game_small.csv  
- Output result: Classifier_Analysis.py and best_trees.txt
- mysklearn folder has all the evaluators and classifiers that are code from scratch
- test_random_forest.py is used to test random forest classifier
- Technical Report.ipynb is a Jupyter Notebook that visualize data and our analysis of the data that we got from sklearn folder.