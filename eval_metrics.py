from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Example data
y_true = [0, 0, 1, 1, 0, 1, 0, 0]
y_pred = [0, 0, 1, 1, 0, 1, 1, 0]

# Calculate Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Calculate Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plotting using seaborn
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            
            xticklabels=['Predicted 0', 
                         'Predicted 1'], 
            
            yticklabels=['Actual 0',
                         'Actual 1'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()