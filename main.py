from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('final_dataset.csv')

# Define preprocessing for numeric and categorical features
numeric_features = ['beds', 'baths', 'size']
categorical_features = ['zip_code']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a pipeline that includes the preprocessor and the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', Ridge())])

# Separate features and target variable for training
X = data[['beds', 'baths', 'size', 'zip_code']]
y = data['price']  # Assuming the target variable is 'price'

# Fit the model
model.fit(X, y)

# Save the pipeline to a file
with open('RidgeModel.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the pipeline from the file
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Convert the data types of the input columns
    input_data['beds'] = pd.to_numeric(input_data['beds'], errors='coerce').fillna(0).astype(int)
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce').fillna(0.0)
    input_data['size'] = pd.to_numeric(input_data['size'], errors='coerce').fillna(0.0)
    input_data['zip_code'] = pd.to_numeric(input_data['zip_code'], errors='coerce').fillna(0).astype(int)

    # Ensure the input data is transformed correctly if the model expects it
    try:
        prediction = pipe.predict(input_data)[0]
    except AttributeError as e:
        print(f"Error during prediction: {e}")
        return str(e), 500

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
