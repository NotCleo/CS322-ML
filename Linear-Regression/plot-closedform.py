import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

PATH_ASSIGNMENT_3 = "/home/amrut/SLR_study_grade_dataset_1.csv"
PATH_ASSIGNMENT_4 = "/home/amrut/SLR_house_price_dataset_2.csv"

def plot_regression():
    print("1. Assignment 3 (Study Hours vs Grades)")
    print("2. Assignment 4 (House Size vs Price)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        file_path = PATH_ASSIGNMENT_3
        x_label_text = "Hours Studied"
        y_label_text = "Grade Obtained"
        title_text = "Assignment 3: Study vs Grades"
    elif choice == '2':
        file_path = PATH_ASSIGNMENT_4
        x_label_text = "House Size (1000 sq.ft)"
        y_label_text = "Price (Lakhs)"
        title_text = "Assignment 4: Size vs Price"

    if not os.path.exists(file_path):
        print(f"\nError: The file was not found at: {file_path}")
        return

    print(f"\n--- Enter weights for {title_text} ---")
    try:
        w0 = float(input("Enter Intercept (w0): ")) 
        w1 = float(input("Enter Slope (w1): "))     
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return

    try:
        data = pd.read_csv(file_path, header=None)
        
   
        data[0] = pd.to_numeric(data[0], errors='coerce')
        data[1] = pd.to_numeric(data[1], errors='coerce')
        
        data = data.dropna()

        X_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values
        
        if len(X_data) == 0:
            print("Error: No valid numeric data found in the CSV.")
            return

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    x_line = np.linspace(min(X_data), max(X_data), 100)
    y_line = w0 + (w1 * x_line)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_data, y_data, color='blue', s=50, label='Actual Data')
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Line: y = {w0:.4f} + {w1:.4f}x')

    plt.title(title_text)
    plt.xlabel(x_label_text)
    plt.ylabel(y_label_text)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()

if __name__ == "__main__":
    plot_regression()
