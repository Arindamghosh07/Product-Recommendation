from flask import Flask, jsonify,  request, render_template
import numpy as np
import pandas as pd
from models import sentiment_based_recommendation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        input = [x for x in request.form.values()]
        print(input[0])
        output = sentiment_based_recommendation(input[0])
        df=pd.DataFrame({'Product':output.index, 'Score':output.values})
        # Display the output of Sentiment Recommendation on the web page
        return render_template('index.html', prediction_text='Recommendations : {} '.format(df.Product))
    else :
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
