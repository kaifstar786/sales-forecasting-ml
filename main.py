
# Sales Forecasting Internship Project
# Predict future sales using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create sample monthly sales data
data = {
    "Month": list(range(1, 13)),
    "Sales": [
        12000, 13500, 12800, 15000,
        16500, 17000, 18500, 19000,
        21000, 22000, 24000, 26000
    ]
}

df = pd.DataFrame(data)

# Save dataset as a CSV file
df.to_csv("sales_data.csv", index=False)

# Train the machine learning model
X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# Predict sales for the next three months
future = pd.DataFrame({"Month": [13, 14, 15]})
predictions = model.predict(future)

print("Future Sales Forecast:")
for month, sales in zip(future["Month"], predictions):
    print(f"Month {month}: Rs. {sales:.2f}")

# Plot historical and forecast sales
plt.plot(df["Month"], y, marker="o", label="Actual Sales")
plt.plot(
    future["Month"],
    predictions,
    marker="x",
    linestyle="--",
    label="Forecast Sales"
)
plt.title("Monthly Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales (Rs.)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the forecast chart
plt.savefig("sales_forecast.png")
plt.show()
