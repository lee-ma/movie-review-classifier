# Movie Review Classifier

This is a Flask webapp that utilizes a model trained via Logistic Regression using Stochastic Gradient Descent. The feature matrices are obtained using HashingVectorizer. The model is continually updated every time the server is restarted based on data collected from user feedback on whether their review was correctly classified.

This project was built following the book: Python Machine Learning 2nd Edition by Sebastian Raschka and Vahid Mirjalili.

## To run locally
1. Clone this repo
2. `pip3 install -r requirements.txt`
3. `python3 app.py`

From there, enter your movie reviews and see if it's classified correctly!
