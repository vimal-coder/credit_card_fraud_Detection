import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
import tkinter as tk
from tkinter import messagebox
df = pd.read_csv("creditcard.csv")
df
df["Class"].value_counts()
legit=df[df.Class==0]
fraud=df[df.Class==1]
print(legit.shape)
print(fraud.shape)
legit.Amount.describe()
fraud.Amount.describe()
df.groupby("Class").mean()
legit_sample=legit.sample(n=492)
new_df=pd.concat([legit_sample ,fraud],axis=0)
x = new_df.drop(columns="Class", axis=1)
y = new_df["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)
# Evaluate the model
x_train_prediction = model.predict(x_train)
training_data_accuracy_score = accuracy_score(x_train_prediction, y_train)
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
print("Accuracy on Training data : ",training_data_accuracy_score )
print("Accuracy score on test data",test_data_accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Create Tkinter window
root = tk.Tk()
root.title("Model Evaluation Results")
# Display results in messagebox
messagebox.showinfo(
    "Model Evaluation Results",
    f"Training Data Accuracy: {training_data_accuracy_score:.2f}\n"
    f"Test Data Accuracy: {test_data_accuracy:.2f}\n"
    f"Confusion Matrix: \n{conf_matrix}\n"
    f"ROC AUC Score: {roc_auc:.2f}"
)
# Function to predict fraud
def predict_fraud():
    try:
        input_features = [float(entry.get()) for entry in feature_entries]
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)
        if prediction == 1:
            result = "Fraudulent"
        else:
            result = "Legitimate"
        messagebox.showinfo("Prediction Result", f"The transaction is {result}.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all features.")

# Create input form for prediction
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

tk.Label(input_frame, text="Enter transaction features:").grid(row=0, columnspan=2)

feature_entries = []
for i in range(x.shape[1]):
    tk.Label(input_frame, text=f"Feature {i+1}:").grid(row=i+1, column=0)
    entry = tk.Entry(input_frame)
    entry.grid(row=i+1, column=1)
    feature_entries.append(entry)

predict_button = tk.Button(root, text="Predict Fraud", command=predict_fraud)
predict_button.pack(pady=10)

# Main loop
root.mainloop()



