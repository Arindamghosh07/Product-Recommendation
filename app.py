from flask import Flask, jsonify,  request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Creating a dataframe out of Sentiments model
sentiments_df = pd.read_csv('reviews.csv' , encoding='latin-1')
#Creating dataframe out of Collaborative model Output
collaborative_df = pd.read_csv('user_final_ratings.csv' , encoding='latin-1')
collaborative_df.set_index(["reviews_username"], inplace=True)

# Defining a Func which will take 20 recommendation based on Collaborative Recommendation model
# and filter out 5 recommendation based on avg sentiment score for those recommendations
def sentiment_based_recommendation(user_input,user_based_final_ratings=collaborative_df,setimentsDF=sentiments_df):

    # Finding the Top 20 products to the user.
    res = user_based_final_ratings.loc[user_input].sort_values(ascending=False)[0:20]
    # Converting recommendation output to a dataframe
    resDF = pd.DataFrame({'name':res.index, 'Score':res.values})
    #Merging the results with our reviews dataset which contains Sentiment score based on Logistic Regression model
    df = pd.merge(resDF, setimentsDF, on=['name'])
    # Filtering top five products having highest sentiment score
    out = df.groupby('name')['Sentiment Prediction'].mean().sort_values(ascending=False)[0:5]
    return out

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
