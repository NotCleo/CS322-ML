#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string.h>
#include <time.h>

#define n 10  
#define d 2

int main(){

    float cost_function_value;
 
    int epoch;
    printf("Enter no of epoch: ");
    scanf("%d", &epoch);
    
    float alpha;
    printf("Enter learning rate: ");
    scanf("%f", &alpha);
    
    srand(time(NULL));
    
    float m = (float)rand() / (float)RAND_MAX;
    float b = (float)rand() / (float)RAND_MAX;
  
    FILE* fptr;
    char line[1024]; 

    fptr = fopen("/home/amrut/SLR_study_grade_dataset_1.csv", "r");

//   just replace with the other csv file and we get that other dataset’s mapping
    
    float input_feature_matrix_x[n][d] = {0};
    float output_feature_matrix_y[n][1] = {0};

    int row = 0;

    while (fgets(line, 1024, fptr) && row < n) {
        input_feature_matrix_x[row][0] = 1.0f;

        char* token = strtok(line, ",");
        int col = 1; 

        while (token != NULL) {
            float value = atof(token); 
            if (col < d) {
                input_feature_matrix_x[row][col] = value;
            } 
            else if (col == d) {
                output_feature_matrix_y[row][0] = value;
            }
            
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    
    for(int k=0; k<epoch; k++)
    {

        float cost_function_value_intermediate = 0; 

        float cost_function_value_derivative_m = 0;
        float cost_function_value_intermediate_m = 0;
        float cost_function_value_derivative_b = 0;
        float cost_function_value_intermediate_b = 0;
    

        for(int t=0; t<n; t++)
        {
            cost_function_value_intermediate += pow(output_feature_matrix_y[t][0] - (m*(input_feature_matrix_x[t][1])+b), 2);    
        }
        cost_function_value = cost_function_value_intermediate / n;
        

        for(int p=0; p<n; p++)
        {

            cost_function_value_intermediate_m += (input_feature_matrix_x[p][1]) * (output_feature_matrix_y[p][0] - (m*(input_feature_matrix_x[p][1])+b));    
        }

        cost_function_value_derivative_m = -2 * (cost_function_value_intermediate_m / n);
        


        for(int j=0; j<n; j++)
        {
            cost_function_value_intermediate_b += (output_feature_matrix_y[j][0] - (m*(input_feature_matrix_x[j][1])+b));    
        }

        cost_function_value_derivative_b = -2 * (cost_function_value_intermediate_b / n);  
        

        m = m - alpha * cost_function_value_derivative_m;
        b = b - alpha * cost_function_value_derivative_b;
        
    }
    
    printf("\nModel Trained.\n");
    printf("The trained slope is %.2f\n", m);
    printf("The trained intercept is %.2f\n", b);
}
