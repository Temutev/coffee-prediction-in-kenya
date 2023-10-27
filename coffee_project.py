import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
coffee_kenya = pd.read_csv("coffee_kenya.csv")
total_sales = pd.read_csv("total_sales.csv")
total_sales.rename(columns={"Year": "YEAR"}, inplace=True)
coffee_data = pd.merge(coffee_kenya, total_sales, on="YEAR", how="inner")
coffee_data['PRODUCTION (Kgs)'] = coffee_data['PRODUCTION (Kgs)'].str.replace(',', '').astype(float)
coffee_data['Total Export Volume (MT)'] = coffee_data['Total Export Volume (MT)'].str.replace(',', '').astype(float)

# Fill missing values
coffee_data['PRODUCTION (Kgs)'].fillna(coffee_data['PRODUCTION (Kgs)'].mean(), inplace=True)
coffee_data.drop(columns=["Total Export Volume (MT)", "Total Export Value(Kshs Billion)","Auction Price(us dollars/50 kg","Yield (Kg/ha)"], inplace=True)
# Standardize the data
scaler = StandardScaler()
X = coffee_data.select_dtypes(include=[int, float]).drop("PRODUCTION (Kgs)", axis=1)
X = scaler.fit_transform(X)
y = coffee_data["PRODUCTION (Kgs)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# load our model
import joblib
model = joblib.load("voting_regressor.joblib")
#model = RandomForestRegressor(random_state=23)
#model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Streamlit app
st.title("Coffee Yield Prediction and Analysis App")

# Section 1: Aim of the Project
st.header("Aim of the Project")
st.write("The primary objective of this project is to develop a predictive model for coffee yield based on a comprehensive analysis of various environmental and agricultural factors. The model's purpose extends beyond prediction; it aims to empower coffee producers, agricultural stakeholders, and policymakers with actionable insights to enhance coffee production in Kenya.")

st.write("Key Goals and Objectives:")
st.write("1. **Accurate Yield Prediction:** The project seeks to create a reliable model that can predict coffee yield based on inputs like rainfall, altitude, temperature, and more. Accurate yield predictions are vital for decision-making and resource allocation in the coffee industry.")

st.write("2. **Optimizing Production:** By understanding the impact of different variables on coffee yield, stakeholders can optimize their agricultural practices. This includes making informed choices regarding planting, harvesting, and post-harvest processing methods.")

st.write("3. **Sustainable Farming:** The project also contributes to sustainable agriculture by exploring the relationship between yield and environmental conditions. It can help promote eco-friendly farming practices in the coffee industry.")

st.write("4. **Data-Driven Insights:** The model generates data-driven insights that can assist coffee producers in decision-making. These insights can range from identifying the most favorable regions for coffee cultivation to improving crop management strategies.")

# Section 2: About the Dataset
st.header("About the Dataset")
st.write("The dataset used for this project is sourced from the Kenya Open Data Initiative and comprises coffee production data spanning the years 2000 to 2005. It provides a valuable resource for understanding the dynamics of coffee production in Kenya during this period.")

st.write("Key Dataset Details:")
st.write("1. **Temporal Scope:** The dataset covers a six-year period, offering a historical perspective on coffee production trends in Kenya.")

st.write("2. **Features:** It includes various features such as rainfall, altitude, temperature, and more, allowing for a multifaceted analysis of the factors influencing coffee yield.")

st.write("3. **Real-World Relevance:** The dataset reflects real-world conditions and challenges faced by coffee producers in Kenya. It encapsulates the effects of changing environmental conditions and market dynamics on coffee production.")

st.write("4. **Sample Data:** Here is a sample of the dataset, demonstrating the variety of features and data points available for analysis:")
st.write(coffee_data.sample(5))

# Section 2: EDA
st.header("Exploratory Data Analysis (EDA)")
st.subheader("Summary Statistics")
st.write(coffee_data.describe())

st.subheader("Data Visualization")
plt.figure(figsize=(10, 6))
sns.scatterplot(x="RAINFALL (mm)", y="PRODUCTION (Kgs)", data=coffee_data)
plt.title("Rainfall vs.Production")
plt.xlabel("Production (Kgs)")
plt.ylabel("Rainfall (mm)")
st.pyplot()
st.write("There is no clear relationship between rainfall and coffee yield.")

plt.figure(figsize=(10, 6))
sns.scatterplot(x="ALTITUTDE (M)", y="PRODUCTION (Kgs)", data=coffee_data)
plt.title("Altitude vs. Production")
plt.xlabel("Production (Kgs)")
plt.ylabel("Altitude (M)")
st.pyplot()
st.write("There is no clear relationship between altitude and coffee yield.")

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Average Temperature (Degree Celsius)", y="Average Rainfall (mm)", data=coffee_data)
plt.title("Average Temperature (Degree Celsius) vs. Average Rainfall (mm)")
plt.xlabel("Average Rainfall (mm)")
plt.ylabel("Average Temperature (Degree Celsius)")
st.pyplot()


st.subheader("Pairwise Correlation Heatmap")
numeric_columns = coffee_data.select_dtypes(include=[int, float]).columns
correlation_matrix = coffee_data[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pairwise Correlation Heatmap")
st.pyplot()

# Section 3: Modelling
st.header("Modeling")
st.subheader("Regression Model")
st.write("The following regression models were trained and evaluated:")
st.write("1. Linear Regression")
st.write("2. Random Forest Regressor")
st.write("3. Gradient Boosting Regressor")
st.write("4. XGBoost Regressor")
st.write("5.Lgbm Regressor")

code ="""
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import Lasso,Ridge
        from math import sqrt


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

        # Define a list of regression models to test
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(random_state=23)),
            ("XGBoost", XGBRegressor(random_state=23)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=23)),
            ("Decision Tree", DecisionTreeRegressor(random_state=23)),
            ("Lasso", Lasso(random_state=23)),
            ("LGBM", LGBMRegressor(random_state=23 ,verbose=-1))

        ]

        # Initialize variables to keep track of the best model and its RMSE
        best_model = None
        best_rmse = float('inf')

        # Loop through the models and evaluate them
        for name, model in models:
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate the root mean squared error (RMSE)
            rmse = sqrt(mean_squared_error(y_test, y_pred, squared=False))

            print(f"{name} RMSE: {rmse}")

            # Update the best model if a better one is found
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse

        print(f"The best model is {best_model.__class__.__name__} with RMSE: {best_rmse}")


        Linear Regression RMSE: 451.5488631987965
        Random Forest RMSE: 426.4242515441051
        XGBoost RMSE: 430.9889924750668
        Gradient Boosting RMSE: 423.57031734517415
        Decision Tree RMSE: 468.38371880357215
        Lasso RMSE: 451.548776176539
        LGBM RMSE: 422.0096440158384
        The best model is LGBMRegressor with RMSE: 422.0096440158384
        """
st.code(code,language="python")

st.write("We then proceeded to use voting regressor to combine the best models and improve the accuracy of the model")

code ="""
#voting regressor
from sklearn.ensemble import VotingRegressor

# Define the base models
base_models = [
    ("Gradient Boosting", GradientBoostingRegressor(random_state=23)),
    ("LGBM", LGBMRegressor(random_state=23 ,verbose=-1)),
]

# Initialize the VotingRegressor
voting_regressor = VotingRegressor(estimators=base_models)
voting_regressor.fit(X_train, y_train)

y_pred = voting_regressor.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred, squared=False))

print(f"Voting Regressor RMSE: {rmse}")

Voting Regressor RMSE: 421.3399485809729
"""

st.code(code,language="python")

st.write("Voting Regressor improved the accuracy of the model to 421.33 which is better than the best model which was LGBMRegressor with RMSE: 422.0096440158384")

st.write("Additionally we used cross validation to improve the accuracy of the model")

code ="""
#cross validation on voting regressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Calculate the cross-validation scores
scores = cross_val_score(voting_regressor, X, y, cv=5, scoring="neg_root_mean_squared_error")

# Convert the scores to positive
scores = abs(scores)

# Calculate the mean score
mean_score = scores.mean()
mean_score = np.sqrt(mean_score)
print(f"Mean RMSE: {mean_score}")
Mean RMSE: 411.4054315721471
"""
st.code(code,language="python")

st.write("Cross validation improved the accuracy of the model to 411.40 which is better than the best model which was LGBMRegressor with RMSE: 422.00")

# Section 4: Best Model and Metrics/Results
st.header("Best Model and Metrics/Results")

st.subheader("Why MSE for Model Evaluation?")
st.write("Mean Squared Error (MSE) is commonly used as an evaluation metric for regression models like the RandomForestRegressor in this project. It measures the average of the squared differences between the predicted values and the actual values. There are several reasons for choosing MSE:")

st.write("1. Sensitivity to Errors: MSE penalizes large errors more than small errors, making it sensitive to outliers. This is important in coffee yield prediction because large errors in production estimation can have significant financial consequences.")

st.write("2. Ease of Interpretation: MSE is easy to interpret. It gives us a sense of how well the model is performing in terms of the variance between predicted and actual values. A lower MSE indicates a better-performing model.")

st.write("3. Mathematical Convenience: MSE is mathematically convenient for optimization. Many machine learning algorithms aim to minimize MSE during training, making it a practical choice.")

st.write("While MSE is a common choice, it's essential to consider the specific context of your problem and the impact of different errors. In some cases, alternative metrics like Mean Absolute Error (MAE) or R-squared may also be valuable for a more comprehensive evaluation.")


st.subheader("Actual vs. Predicted")

code ="""
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with 'Actual' and 'Predicted' columns
# You can create a bar plot to compare actual and predicted values

plt.figure(figsize=(10, 6))

# Create the x-axis values (observation numbers)
x_values = range(len(df))

# Plot the actual and predicted values as bars
plt.bar(x_values, df['Actual'], width=0.4, label='Actual', align='center', color='blue', alpha=0.7)
plt.bar(x_values, df['Predicted'], width=0.4, label='Predicted', align='edge', color='green', alpha=0.7)

plt.title("Actual vs. Predicted")
plt.xlabel("Observation")

plt.legend()
plt.show()

"""
st.code(code,language="python")
st.bar_chart(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# Section 5: Further Research
st.header("Further Research")

st.subheader("1. Data Enrichment and Feature Expansion")
st.write("One way to enhance this project is by collecting more comprehensive and granular data related to coffee production. This could include additional features such as soil quality, pest and disease monitoring, crop management practices, and the use of fertilizers and pesticides. By expanding the dataset, we can gain a deeper understanding of the factors influencing coffee yield, leading to more accurate predictions.")


st.subheader("2. Integration of Real-Time Weather Data")
st.write("To enhance prediction accuracy, consider integrating real-time weather data into the model. Weather conditions significantly impact coffee yield, and having up-to-date weather information can lead to more precise predictions. This can involve API integrations to fetch current weather data and incorporating it into the model's input features.")

st.subheader("3. Sustainable Agricultural Practices and Their Impact")
st.write("Research the influence of sustainable agricultural practices on coffee yield. Sustainable practices, such as organic farming, shade-grown coffee, and environmentally friendly cultivation methods, have gained attention in recent years. Analyzing their impact on coffee production can provide insights into environmentally conscious and economically viable approaches to coffee farming.")

st.write("By exploring these areas of research, we can not only improve the predictive power of our model but also contribute to a more sustainable and efficient coffee industry, benefitting both coffee producers and consumers.")

# Section 6: Try the Model
st.header("Try the Model")
st.write("You can interact with the model by providing custom input values for various features to get predictions. This feature allows you to experiment with different scenarios and see how changes in factors affect coffee yield.")

year = st.number_input("Year", value=2005, min_value=2000, max_value=2005)
rainfall = st.number_input("Rainfall (mm)", min_value=1000, max_value=1600)
altitude = st.number_input("Altitude (M)", min_value=1255, max_value=1837)
avg_rainfall = st.number_input("Average Rainfall (mm)", min_value=498.41, max_value=814.71)
avg_temp = st.number_input("Average Temperature (Degree Celsius)", min_value=24.93, max_value=25.38)

if st.button("Predict Coffee Yield"):
    custom_data = pd.DataFrame({
        "YEAR": [year],
        "RAINFALL (mm)": [rainfall],
        "ALTITUTDE (M)": [altitude],
        "Average Rainfall (mm)": [avg_rainfall],
        "Average Temperature (Degree Celsius)": [avg_temp]
    })
    custom_data = scaler.transform(custom_data)
    prediction = model.predict(custom_data)
    st.success(f"Predicted Coffee Yield: {prediction[0]:.2f}")

