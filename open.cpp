#include <bits/stdc++.h>
using namespace std;

string return_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/predict_1.txt";

void return_text(vector<string>& predict, string return_file){
    ofstream file(return_file);
    if(!file.is_open()){
        cout << "Error!";
        return;
    }
    for(int i = 0; i < predict.size(); i++){
        file << predict[i];
        file << "\n";
    }
    file.close();
}

int main(){
    vector<string> tmp;
    tmp.push_back("abc");
    tmp.push_back("cba");
    tmp.push_back("bac");
    return_text(tmp, return_file);
}