#include <bits/stdc++.h>
using namespace std;

string train_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/train.csv";
string test_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/test.csv";

void extract_csv(vector<vector<double>>& features, vector<string>& labels, string filename);
struct Node;
double gini_score(vector<string>& labels);
double calc_total_gini(vector<string>& left_gini, vector<string>& right_gini);
pair<double, double> find_best_split(vector<vector<double>>& features, vector<string>& labels, int feature);
Node* build(vector<vector<double>>& features, vector<string>& labels, int depth = 0);
string predict(Node* root, vector<double>& samples);

void extract_csv(vector<vector<double>>& features, vector<string>& labels, string filename){
    ifstream file(filename);
    int bit;
    if(filename == train_file) bit = 1;
    else bit = 0;
    if(!file.is_open()){
        cout << "Error!" << endl;
        return;
    }
    string line;
    while(getline(file, line)){
        stringstream ss(line);
        string value;
        if(bit == 1){
            getline(ss, value, ',');
            labels.push_back(value);
        }
        vector<double> feature;
        while(getline(ss, value, ',')){
            feature.push_back(stod(value));
        }
        features.push_back(feature);
    }
}

struct Node {
    string label;
    int feature;
    double threshold;
    Node* left;
    Node* right;

    Node(string label): label(label), feature(-1), threshold(0), left(nullptr), right(nullptr) {}
    Node(int feature, double threshold): label(""), feature(feature), threshold(threshold), left(nullptr), right(nullptr) {}
};

double gini_score(vector<string>& labels){
    unordered_map<string, int> labelfre;
    for(auto s : labels) labelfre[s]++;
    double gini = 1.0;
    for(auto i : labelfre){
        double prob = ((double)i.second)/((double) labels.size());
        gini -= prob*prob;
    }
    return gini;
}

double calc_total_gini(vector<string>& left_gini, vector<string>& right_gini){
    int total_size = left_gini.size() + right_gini.size();
    return ((double)left_gini.size()*gini_score(left_gini) + (double)right_gini.size()*gini_score(right_gini))/((double) total_size);
}

pair<double, double> find_best_split(vector<vector<double>>& features, vector<string>& labels, int feature){
    unordered_set<double> featureSet;
    for(int i = 0; i < features.size(); i++){
        featureSet.insert(features[i][feature]);
    }
    double best_threshold = -1;
    double best_gini_score = 100000000.0;
    for(double threshold : featureSet){
        vector<string> left_gini;
        vector<string> right_gini;
        for(int i = 0; i < features.size(); i++){
            if(features[i][feature] <= threshold) left_gini.push_back(labels[i]);
            else right_gini.push_back(labels[i]);
        }
        double total_gini = calc_total_gini(left_gini, right_gini);
        if(total_gini < best_gini_score){
            best_gini_score = total_gini;
            best_threshold = threshold;
        }
    }
    return {best_gini_score, best_threshold};
}

Node* build(vector<vector<double>>& features, vector<string>& labels, int depth){
    unordered_set<string> labelSet(labels.begin(), labels.end());
    if(labelSet.size() == 1){
        return new Node(*labelSet.begin());
    }
    int best_feature = -1;
    double best_threshold = -1;
    double best_gini_score = 100000000.0;
    for(int feature = 0; feature < features[0].size(); feature++){
        auto best_split = find_best_split(features, labels, feature);
        double gini = best_split.first;
        double threshold = best_split.second;
        if(gini < best_gini_score){
            best_gini_score = gini;
            best_threshold = threshold;
            best_feature = feature;
        }
    }

    if(best_feature == -1){
        map<string, int> cnt;
        for(auto s : labels){
            cnt[s]++;
        }
        string res = max_element(cnt.begin(), cnt.end(), [](const auto& a, const auto& b){
            return a.second < b.second;
        })->first;

        return new Node(res);
    }

    vector<vector<double>> left_features, right_features;
    vector<string> left_label, right_label;
    for(int i = 0; i < features.size(); i++){
        if(features[i][best_feature] <= best_threshold){
            left_features.push_back(features[i]);
            left_label.push_back(labels[i]);
        }
        else{
            right_features.push_back(features[i]);
            right_label.push_back(labels[i]);
        }
    }

    Node* node = new Node(best_feature, best_threshold);
    node->left = build(left_features, left_label, depth+1);
    node->right = build(right_features, right_label, depth+1);

    return node;
}

string predict(Node* root, vector<double>& samples){
    if(!root->label.empty()){
        return root->label;
    }
    if(samples[root->feature] <= root->threshold){
        return predict(root->left, samples);
    }
    else return predict(root->right, samples);
}

int main(){
    vector<vector<double>> train_features;
    vector<string> train_labels;
    vector<vector<double>> test_features;
    vector<string> test_labels;
    extract_csv(train_features, train_labels, train_file);
    extract_csv(test_features, test_labels, test_file);
    Node* decision_tree = build(train_features, train_labels, 0);
    for(int i = 0; i < test_features.size(); i++){
        vector<double> features = test_features[i];
        test_labels.push_back(predict(decision_tree, features));
    }
    cout << test_labels.size() << endl;
    for(auto s : test_labels) cout << s << endl;
    // for(auto s : train_labels) cout << s << endl;
}