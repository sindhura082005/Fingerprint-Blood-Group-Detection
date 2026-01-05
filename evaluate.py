import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
model = load_model("blood_group_cnn_model.h5")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    directory="dataset",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

class_labels = list(val_generator.class_indices.keys())
report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels
)

print("\nClassification Report:\n")
print(report)
with open("results/classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()
