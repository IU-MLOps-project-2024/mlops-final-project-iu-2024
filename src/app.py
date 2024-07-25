
import gradio as gr
import requests
import json
import pickle
import pandas as pd
import numpy as np

# from data import preprocess_data

# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder

scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
vectorizer_item_description = pickle.load(open('vectorizer_item_description.pkl', 'rb'))
vectorizer_item_name = pickle.load(open('vectorizer_item_name.pkl', 'rb'))
vectorizer_item_variation = pickle.load(open('vectorizer_item_variation.pkl', 'rb'))

def preprocess_data(df):

    # get rid of nan values
    df.loc[df['item_name'].isna(), 'item_name'] = ""
    df.loc[df['item_description'].isna(), 'item_description'] = ""
    df.loc[df['item_variation'].isna(), 'item_variation'] = ""
    
    df[['price', 'stock']] = scaler.transform(df[['price', 'stock']].to_numpy())

    # break down item creation date
    df['item_creation_date'] = pd.to_datetime(df['item_creation_date'])
    df['year'] = df['item_creation_date'].dt.year
    df['month'] = df['item_creation_date'].dt.month
    df['day'] = df['item_creation_date'].dt.day
    df.drop(columns=['item_creation_date'], inplace=True)

    # prepare X and y datasets
    numerical_features = [
        'itemid', 'shopid', 'price', 'stock', 'cb_option',
        'is_preferred', 'sold_count', 'year', 'month', 'day'
    ]
    X = df[numerical_features].to_numpy()

    X = np.concatenate(
        (
            X,
            vectorizer_item_name.transform([df.loc[0, 'item_name']]).toarray(),
            vectorizer_item_description.transform([df.loc[0, 'item_description']]).toarray(),
            vectorizer_item_variation.transform([df.loc[0, 'item_variation']]).toarray()
        ),
        axis=1
    )

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])

    return X

# Define the function to get predictions from the Flask API
def predict(
    itemid,
    shopid,
    item_name,
    item_description,
    item_variation,
    price,
    stock,
    cb_option,
    is_preferred,
    sold_count,
    item_creation_date,
):
    
    raw_df = pd.DataFrame({
        'itemid': [itemid],
        'shopid': [shopid],
        'item_name': [item_name],
        'item_description': [item_description],
        'item_variation': [item_variation],
        'price': [price],
        'stock': [stock],
        'cb_option': [cb_option],
        'is_preferred': [is_preferred],
        'sold_count': [sold_count],
        'item_creation_date': [item_creation_date]
    })

    X = preprocess_data(raw_df)

    url = "http://localhost:5001/predict"
    input_data = {"features": X.iloc[0, :].to_list()}
    response = requests.post(url, json=input_data)
    
    # print(response.response)

    if response.status_code == 200:
        prediction = response.json()["predictions"]
        category_class = np.argmax(prediction, axis=1)
        category = encoder.inverse_transform(category_class)[0]
        return category
    else:
        return f"Error: {response.json()['error']}"

demo = gr.Interface(
    # The predict function will accept inputs as arguments and return output
    fn=predict,
    
    # Here, the arguments in `predict` function
    # will populated from the values of these input components
    inputs = [
        # Select proper components for data types of the columns in your raw dataset
        gr.Number(label="itemid", value=20046620), 
        gr.Number(label="shopid", value=760000), 
        gr.Text(label="item_name", value="101% AUTHENTIC BASEBALL CAPS "),
        gr.Text(label="item_description", value="PREORDER Takes about 23 weeks to arrive, will provide receipt of order & estimated date of arrival upon confirmationAny other designs youre looking for feel free to ask! Ill help you check the website to see if i am able to ship it for you (: Sold more than 10 caps with good reviews in Carousell, 100% authentic and can be trusted. From a denmark store which is an official rattle location for yeezyboost Check listings for other authentic baseball caps too! #Nike #baseball #baseballcap #newera #neweracap #authentic #swoosh #newyork #yankee"),
        gr.Text(label="item_variation", value="{NEWERA BLACK: 35.0, NIKE SWOOSH DENIM: 35.0, NIKE SWOOSH BLACK: 35.0, NIKE SMALL SWOOSH: 35.0, NEWERA WHITE: 35.0, NEWERA MAROON: 35.0}"),
        gr.Number(label="price", value=35),
        gr.Number(label="stock", value=300), 
        gr.Dropdown(label="cb_option", choices=[0, 1]),
        gr.Dropdown(label="is_preferred", choices=[0, 1]),  
        gr.Number(label="sold_count", value=0), 
        gr.Text(label="item_creation_date", value="9/5/16 1:14")
    ],
    
    # The outputs here will get the returned value from `predict` function
    outputs = gr.Text(label="category"),
)


# Launch the Gradio app
demo.launch()