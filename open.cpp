#include <bits/stdc++.h>
using namespace std;

int main(){
    string filename = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/train.csv";
    ifstream file(filename);
    if(!file.is_open()){
        cout << "error!";
        return 1;
    }
    vector<string> train_labels;
    vector<vector<double>> train_features;
    string line;
    while(getline(file, line)){
        stringstream ss(line);
        string s;
        getline(ss, s, ',');
        train_labels.push_back(s);
        vector<double> features;
        while(getline(ss, s, ',')){
            features.push_back(stod(s));
        }
        train_features.push_back(features);
    }
    file.close();
    for(int i = 0; i < train_labels.size(); i++){  
        cout << train_labels[i] << " ";
        for(int j = 0; j < train_features[i].size(); j++){
            cout << train_features[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}