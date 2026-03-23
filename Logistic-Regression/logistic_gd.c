#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>

#define MAX_ROWS 150  
#define d 3 // 1 Intercept + 2 Features

float sigmoid(float z) {
    return 1.0f / (1.0f + exp(-z));
}

int main(){
    int epoch;
    printf("Enter no of epochs (e.g., 10000): ");
    if (scanf("%d", &epoch) != 1) return 1;
    
    float alpha;
    printf("Enter learning rate (e.g., 0.01): ");
    if (scanf("%f", &alpha) != 1) return 1;
  
    FILE *fx = fopen("/home/amrut/Downloads/logisticX.csv", "r");
    FILE *fy = fopen("/home/amrut/Downloads/logisticY.csv", "r");

    if (!fx || !fy) {
        printf("Error: Could not open dataset files.\n");
        return 1;
    }
    
    float X[MAX_ROWS][d] = {0};
    float y[MAX_ROWS][1] = {0};

    int m_count = 0;
    char lineX[1024], lineY[1024]; 

    // Preprocessing: Read X and Y, add intercept x0 = 1
    while (fgets(lineX, 1024, fx) && fgets(lineY, 1024, fy) && m_count < MAX_ROWS) {
        X[m_count][0] = 1.0f; // Intercept term

        char* token = strtok(lineX, ",");
        X[m_count][1] = atof(token); // Feature 1
        
        token = strtok(NULL, ",");
        if (token != NULL) {
            X[m_count][2] = atof(token); // Feature 2
        }

        y[m_count][0] = atof(lineY); // Target label
        m_count++;
    }
    fclose(fx); fclose(fy);
    printf("Successfully loaded %d rows.\n", m_count);
    
    float theta[d] = {0}; 
    
    for(int k = 0; k < epoch; k++) {
        float grad[d] = {0};

        for(int i = 0; i < m_count; i++) {
            float z = 0;
            for(int j = 0; j < d; j++) {
                z += theta[j] * X[i][j];
            }
            
            float h = sigmoid(z);
            float error = h - y[i][0];
            
            for(int j = 0; j < d; j++) {
                grad[j] += error * X[i][j];
            }
        }

        for(int j = 0; j < d; j++) {
            theta[j] -= alpha * (grad[j] / m_count);
        }
    }
    
    printf("\nGD Model Trained.\n");
    printf("Theta 0 (Intercept): %f\n", theta[0]);
    printf("Theta 1: %f\n", theta[1]);
    printf("Theta 2: %f\n\n", theta[2]);
    
    // ==========================================
    // 1. Calculate Overall Training Accuracy
    // ==========================================
    int correct_predictions = 0;
    for (int i = 0; i < m_count; i++) {
        float z = theta[0] + theta[1] * X[i][1] + theta[2] * X[i][2];
        float predicted_prob = sigmoid(z);
        int predicted_class = (predicted_prob >= 0.5f) ? 1 : 0;
        
        if (predicted_class == (int)y[i][0]) {
            correct_predictions++;
        }
    }
    float accuracy = ((float)correct_predictions / m_count) * 100.0f;
    printf("Training Accuracy: %.2f%%\n\n", accuracy);
// ==========================================
    // 2. Interactive Prediction Loop
    // ==========================================
    int choice;
    float input_x1, input_x2;

    while(1) {
        printf("1. Predict Class for new data\n");
        printf("2. Exit\n");
        printf("Choice: ");
        
        if (scanf("%d", &choice) != 1) {
            while(getchar() != '\n'); // clear buffer
            continue;
        }

        if (choice == 1) {
            printf("Enter Feature 1 (x1): ");
            if (scanf("%f", &input_x1) != 1) {
                while(getchar() != '\n'); 
                printf("Invalid input. Please enter a number.\n\n");
                continue;
            }
            
            printf("Enter Feature 2 (x2): ");
            if (scanf("%f", &input_x2) != 1) {
                while(getchar() != '\n'); 
                printf("Invalid input. Please enter a number.\n\n");
                continue;
            }

            float z = theta[0] + (theta[1] * input_x1) + (theta[2] * input_x2);
            float prob = sigmoid(z);
            int pred_class = (prob >= 0.5f) ? 1 : 0;

            printf("\n--- Results ---\n");
            printf("Z value: %.4f\n", z);
            printf("Predicted Probability: %.4f\n", prob);
            printf("Predicted Class: %d\n\n", pred_class);
        } 
        else if (choice == 2) {
            printf("Exiting...\n");
            break;
        } 
        else {
            printf("Invalid choice. Try again.\n\n");
        }
    }    return 0;
}
