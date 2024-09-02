import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



class steering_wheel_model():
    def __init__(self):
        folder_path = './data/'
        exclude_list = ['HOD_DATA1.xlsx', 'HOD_DATA_#Uc120#Uc6b0.xlsx']
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.xlsx') and f not in exclude_list]
        
        data_list = []
        for file in file_names:
            data_list.append(pd.read_excel('./data/' + file))
            
        all_data = pd.concat(data_list, ignore_index=True)

        self.X = all_data[['Sensor', 'Shield', 'Sensor_changes', 'Shield_changes', 'Diff', 'Ratio']]
        self.y = all_data[['Hand-on', 'Touch Type', 'Touch Material']]  # 단일 라벨 'Result'만 사용

    def get_data(self):
        return self.X, self.y
    
    def train_model(self, model):
        model = model
        accuracies = {}

        for label in self.y.columns:
            print(f"Evaluating label: {label}")

            y_label = self.y[label]

            X_train, X_test, y_train, y_test = train_test_split(self.X, y_label, test_size=0.3, random_state=42)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            accuracies[label] = accuracy

            print(f"Accuracy for {label}: {accuracy:.4f}")

        # 평균 정확도 계산
        average_accuracy = sum(accuracies.values()) / len(accuracies)
        print(f"Average Accuracy: {average_accuracy:.4f}")


if __name__ == "__main__":
    trainer = steering_wheel_model()

    model = LogisticRegression(max_iter=700, random_state=42)
    trainer.train_model(model)