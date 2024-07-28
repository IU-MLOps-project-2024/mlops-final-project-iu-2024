import mlflow
import giskard
import pandas as pd
import numpy as np
from model import retrieve_model_with_alias
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle



df = pd.read_csv("data/raw/Test_Pandas.csv")
df.dropna(inplace=True)
df['item_creation_date'] = pd.to_datetime(df['item_creation_date'])
df = df.iloc[int(len(df) * 0.8):len(df)]

giskard_dataset = giskard.Dataset(
    df=df,
    target='category'
)

def preprocess_data(df):
    def vectorize_text(vectorizer, text):
        return vectorizer.transform(text).toarray()

    # get rid of nan values
    df.loc[df['item_name'].isna(), 'item_name'] = ""
    df.loc[df['item_description'].isna(), 'item_description'] = ""
    df.loc[df['item_variation'].isna(), 'item_variation'] = ""

    # scale continuous values
    scaler = pickle.load(open("./scaler.pkl", "rb"))
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
            vectorize_text(pickle.load(open("./vectorizer_item_name.pkl", "rb")), df['item_name']),
            vectorize_text(pickle.load(open("./vectorizer_item_description.pkl", "rb")), df['item_description']),
            vectorize_text(pickle.load(open("./vectorizer_item_variation.pkl", "rb")), df['item_variation'])
        ),
        axis=1
    )

    scaler2 = pickle.load(open("./scaler2.pkl", "rb"))
    X = scaler2.transform(X)
    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])], dtype=np.float32)

    return X

# Load the best model (replace 'model' with your actual model path if different)
model: mlflow.pyfunc.PyFuncModel = retrieve_model_with_alias("Transformer", "champion")

def predict(df):
    X = preprocess_data(df)
    return model.predict(X)

encoder = LabelEncoder()

with open('./encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

labels = encoder.classes_

# Validate the model using Giskard
giskard_model = giskard.Model(
  model=predict,
  model_type = "classification",
  classification_labels=labels
)

model_name = "MLP"
model_alias = "champion"

scan_results = giskard.scan(giskard_model, giskard_dataset)
scan_results_path = f"./reports/validation_results_{model_name}_{model_alias}.html"
scan_results.to_html(scan_results_path)

suite_name = f"test_suite_{model_name}_{model_alias}"
test_suite = giskard.Suite(name = suite_name)

test1 = giskard.testing.test_f1(model = giskard_model,
                                dataset = giskard_dataset,
                                threshold=0.3)

test_suite.add_test(test1)

test_results = test_suite.run()
if (test_results.passed):
    print("Passed model validation!")
else:
    print("Model has vulnerabilities!")
