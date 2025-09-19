import os
import subprocess

def run_preprocessing():
    print("\n[1] Running Data Preprocessing...")
    subprocess.run(["python", "scripts/data_preprocessing.py"])

def run_model_training():
    print("\n[2] Running Model Training and Comparison...")
    subprocess.run(["python", "scripts/model_training.py"])

def run_confusion_matrix():
    print("\n[3] Running Confusion Matrix Visualizer...")
    subprocess.run(["python", "scripts/confusion_matrix.py"])

# def run_script():
#     print("\n[4] Running Custom Script...")
#     subprocess.run(["python", "scripts/script.py"])

def menu():
    while True:
        print("\n========= Injury Prediction Project Menu =========")
        print("1. Run Preprocessing")
        print("2. Train and Compare Models")
        print("3. Visualize Confusion Matrix")
        # print("4. Run Custom Script")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            run_preprocessing()
        elif choice == '2':
            run_model_training()
        elif choice == '3':
            run_confusion_matrix()
        # elif choice == '4':
        #     run_script()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    menu()
