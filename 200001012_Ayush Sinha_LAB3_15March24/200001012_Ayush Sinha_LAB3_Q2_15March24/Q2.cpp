#include <bits/stdc++.h>
using namespace std;
//assuming equiprobable probablity as 1/4 (pi(s,a) = 1);
int helper(int i, int j, double arr[][4]){
    if(i == 4) return arr[i-1][j];
    if(i == -1)return arr[i+1][j];
    if(j == 4) return arr[i][j-1];
    if(j == -1)return arr[i][j+1];
    return arr[i][j];
    
}
int main(){
    double arr[4][4];//val arr
    for(int i= 0; i<4; i++){
        for(int j = 0; j<4; j++){
            arr[i][j] = 0.0;
        }
    }
    int r = -1;

    double epsilon = 0.5;
    double maxi = 1e9;
    double newArr[4][4];
    int count = 10;
    while(maxi > epsilon && count --){
        maxi = 0;
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                if((i == 3 && j == 3) || (i == 0 && j == 0)) {
                    newArr[i][j] = 0;
                    continue;
                }
                newArr[i][j] = arr[i][j]; // Copy the current value to new array
                newArr[i][j] += (0.25)*(-4.0 + 0.1*(helper(i,j+1,arr) + helper(i,j-1,arr) + helper(i+1,j,arr) + helper(i-1,j,arr)));
                double oldValue = arr[i][j]; // Store the old value
                // newArr[i][j] = (0.25) * (-4.0 + 0.1 * (helper(i, j + 1, arr) + helper(i, j - 1, arr) + helper(i + 1, j, arr) + helper(i - 1, j, arr)));
                double change = abs(oldValue - newArr[i][j]); // Calculate change
                maxi = max(maxi, change); // Update maxi if change is larger
            }
        }

        for(int i= 0; i<4; i++){
            for(int j = 0; j<4; j++){
                arr[i][j] = newArr[i][j];
            }
        }
    }
    for(int i = 0; i<4; i++){
        for(int j = 0; j<4; j++){
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}