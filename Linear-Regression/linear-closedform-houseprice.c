#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>

#define n 10  
#define d 2

void inversematrix(float matrix[d][d], float inverse[d][d]) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            if (i == j) inverse[i][j] = 1.0;
            else inverse[i][j] = 0.0;
        }
    }

    for (int i = 0; i < d; i++) {
        int pivot = i;
        for (int j = i + 1; j < d; j++) {
            if (fabs(matrix[j][i]) > fabs(matrix[pivot][i])) {
                pivot = j;
            }
        }
        
        float temp;
        for (int k = 0; k < d; k++) {
            temp = matrix[i][k];
            matrix[i][k] = matrix[pivot][k];
            matrix[pivot][k] = temp;
            
            temp = inverse[i][k];
            inverse[i][k] = inverse[pivot][k];
            inverse[pivot][k] = temp;
        }

        if (fabs(matrix[i][i]) < 1e-9) {
            printf("The matrix is singular\n");
            exit(1);
        }

        float divisor = matrix[i][i];
        for (int j = 0; j < d; j++) {
            matrix[i][j] /= divisor;
            inverse[i][j] /= divisor;
        }

        for (int k = 0; k < d; k++) {
            if (k != i) {
                float factor = matrix[k][i];
                for (int j = 0; j < d; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                    inverse[k][j] -= factor * inverse[i][j];
                }
            }
        }
    }
}

int main(){

    FILE* fptr;
    char line[1024]; 

    fptr = fopen("/home/amrut/SLR_house_price_dataset_2.csv", "r");
    
    float X[n][d] = {0};
    float y[n][1] = {0};

    int row = 0;
    

    while (fgets(line, 1024, fptr) && row < n) {
        X[row][0] = 1.0f;

        char* token = strtok(line, ",");
        int col = 1; 

        while (token != NULL) {
            float value = atof(token); 
            if (col < d) {
                X[row][col] = value;
            } 
            else if (col == d) {
                y[row][0] = value;
            }
            
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }

    fclose(fptr); 
    printf("Successfully loaded %d rows.\n", row);


    float X_transpose[d][n];
    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            X_transpose[j][i] = X[i][j];            
        }
    }

    float XtX[d][d];
    for(int i=0; i<d; i++){
        for(int j=0; j<d; j++){
            float sum = 0;
            for(int k=0; k<n; k++){
                sum += X_transpose[i][k] * X[k][j];                        
            }
            XtX[i][j] = sum;
        }
    }
    
    float XtX_Inverse[d][d];
    inversematrix(XtX, XtX_Inverse);

    float Xty[d][1];
    for(int i=0; i<d; i++){
        for(int j=0; j<1; j++){
            float sum = 0; 
            for(int k=0; k<n; k++){
                sum += X_transpose[i][k] * y[k][j];                        
            }
            Xty[i][j] = sum;
        }
    }
    
    float weights[d][1]; 
    for(int i=0; i<d; i++){
        for(int j=0; j<1; j++){
            float sum = 0;
            for(int k=0; k<d; k++){ 
                sum += XtX_Inverse[i][k] * Xty[k][j];                        
            }
            weights[i][j] = sum;
        }
    }   
    
    printf("\nModel Trained.\n");
    printf("y = %.4f + %.4f * x\n\n", weights[0][0], weights[1][0]);

    int choice;
    float input_val;
    float prediction;

    while(1) {
        printf("1. Predict Grade\n");
        printf("2. Exit\n");
        printf("Choice: ");
        scanf("%d", &choice);

        if (choice == 1) {
            printf("Enter Hours Studied: ");
            scanf("%f", &input_val);

            prediction = weights[0][0] + (weights[1][0] * input_val);

            printf("Predicted Grade: %.3f\n\n", prediction);
        } 
        else if (choice == 2) {
            break;
        } 

    }
}
