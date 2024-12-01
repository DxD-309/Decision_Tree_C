#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> confusion_matrix(vector<string>& pred, vector<string>& label){
    vector<vector<int>> matrix(3, vector<int>(3, 0));
    vector<int> pred_int;
    vector<int> label_int;
    for(unsigned int i = 0; i < pred.size(); i++){
        if(pred[i] == "L") pred_int.push_back(0);
        if(pred[i] == "R") pred_int.push_back(1);
        if(pred[i] == "B") pred_int.push_back(2);

        if(label[i] == "L") label_int.push_back(0);
        if(label[i] == "R") label_int.push_back(1);
        if(label[i] == "B") label_int.push_back(2);
    }
    for(unsigned int i = 0; i < pred.size(); i++){
        int pred_class = pred_int[i];
        int true_class = label_int[i];
        matrix[pred_class][true_class]++;
    }

    return matrix;
}

double precision_score(vector<vector<int>>& confusion_matrix, int i){
    double tp, fp;
    tp = (double) confusion_matrix[i][i];
    if(i == 0){
        fp = (double) confusion_matrix[0][1] + (double) confusion_matrix[0][2];
    }
    else if(i == 1){
        fp = (double) confusion_matrix[1][0] + (double) confusion_matrix[1][2];
    }
    else fp = (double) confusion_matrix[2][0] + (double) confusion_matrix[2][1];

    return tp/(tp+fp);
}

double recall_score(vector<vector<int>>& confusion_matrix, int i){
    double tp, fn;
    tp = (double) confusion_matrix[i][i];
    if(i == 0){
        fn = (double)confusion_matrix[1][0] + (double) confusion_matrix[2][0];
    }
    else if(i == 1){
        fn = (double) confusion_matrix[0][1] + (double) confusion_matrix[2][1];
    }
    else fn = (double) confusion_matrix[0][2] + (double) confusion_matrix[1][2];

    return tp/(tp+fn);
}

double f1_score(vector<vector<int>>& confusion_matrix, int i){
    double precision = precision_score(confusion_matrix, i);
    double recall = recall_score(confusion_matrix, i);
    if(precision + recall == 0) return 0.0;
    return (2*precision*recall)/(precision+recall);
}

double f1_macro(vector<vector<int>>& confusion_matrix){
    double F1_L = f1_score(confusion_matrix, 0);
    double F1_R = f1_score(confusion_matrix, 1);
    double F1_B = f1_score(confusion_matrix, 2);
    return (F1_L+F1_R+F1_B)/3.0;
}

int main(){
    vector<string> pred;
    vector<string> label;
    pred.push_back("L");
    pred.push_back("L");
    pred.push_back("L");
    pred.push_back("R");
    pred.push_back("R");
    pred.push_back("R");
    pred.push_back("B");
    pred.push_back("B");
    pred.push_back("B");    

    label.push_back("L");
    label.push_back("L");
    label.push_back("B");
    label.push_back("L");
    label.push_back("R");
    label.push_back("R");
    label.push_back("B");
    label.push_back("L");
    label.push_back("R");

    vector<vector<int>> confusion_matrixx = confusion_matrix(pred, label);
    double f1_tmp = f1_macro(confusion_matrixx);
    cout << f1_tmp << endl;
    // for(int i = 0; i < 3; i++){
    //     for(int j = 0; j < 3; j++){
    //         cout << confusion_matrixx[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    return 0;
}   