#include <bits/stdc++.h>
using namespace std;

string train_file = "C:/Users/Admin/VSCODE/test/nmlt/Decision_Tree_C/train.csv";
string test_file = "C:/Users/Admin/VSCODE/test/nmlt/Decision_Tree_C/test.csv";

void extract_csv(vector<vector<double>>& features, vector<string>& labels, string filename);
struct Node;
double gini_score(vector<string>& labels);
double calc_total_gini(vector<string>& left_gini, vector<string>& right_gini);
pair<double, double> find_best_split(vector<vector<double>>& features, vector<string>& labels, int feature);
Node* build(vector<vector<double>>& features, vector<string>& labels, int depth, int leaf, int sample_split, int sample_leaf,
            int max_depth, int max_leaf, int min_sample_split, int min_sample_leaf);
string predict(Node* root, vector<double>& samples);
double f1_macro(vector<string>& pred, vector<string>& test_label);
double f1_micro(vector<string>& pred, vector<string>& test_label)
double cross_validation(vector<vector<double>>& features, vector<string>& labels, int k, int f1_type, int max_depth, 
                        int max_leaf, int min_sample_split, int min_sample_leaf);

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

Node* build(vector<vector<double>>& features, vector<string>& labels, int depth, int leaf, int sample_split, int sample_leaf,
            int max_depth, int max_leaf, int min_sample_split, int min_sample_leaf){
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

    if(best_feature == -1 || depth == max_depth || leaf == max_leaf || sample_split < min_sample_split || sample_leaf < min_sample_leaf){
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
    node->left = build(left_features, left_label, depth+1, leaf*2, left_features.size(), left_features.size(), max_depth, 
                        max_leaf, min_sample_split, min_sample_leaf);
    node->right = build(right_features, right_label, depth+1, leaf*2+1, right_features.size(), right_features.size(), max_depth, 
                        max_leaf, min_sample_split, min_sample_leaf);

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

double f1_macro(vector<string>& pred, vector<string>& test_label){

}

double f1_micro(vector<string>& pred, vector<string>& test_label){

}

double cross_validation(vector<vector<double>>& features, vector<string>& labels, int k, string f1_type, int max_depth, 
                        int max_leaf, int min_sample_split, int min_sample_leaf){
    int fold_size = features.size()/k;
    for(int i = 0; i < k; i++){
        vector<vector<double>> train_feature, test_feature;
        vector<string> train_label, test_label;

        for(int j = 0; j < features.size(); j++){
            if(j/fold_size == i){
                test_feature.push_back(features[j]);
                test_label.push_back(labels[j]);
            }
            else{
                train_feature.push_back(features[j]);
                train_label.push_back(labels[j]);
            }
        }

        Node* decision_tree = build(train_feature, train_label, 0, 1, train_feature.size(), train_feature.size(),
                                    max_depth, max_leaf, min_sample_split, min_sample_leaf);

        vector<string> pred;
        for(int i = 0; i < test_feature.size(); i++){
            string tmp = predict(decision_tree, test_feature[i]);
            pred.push_back(tmp);
        }

        double f1_score = 0;
        if(f1_type == "macro"){
            f1_score += f1_macro(pred, test_label);
        }
        else{
            f1_score += f1_micro(pred, test_label);
        }
        return f1_score/k;
    }
}

int main(){
    vector<vector<double>> train_features;
    vector<string> train_labels;
    vector<vector<double>> test_features;
    vector<string> test_labels;
    extract_csv(train_features, train_labels, train_file);
    extract_csv(test_features, test_labels, test_file);
    Node* decision_tree = build(train_features, train_labels, 0, 1, train_features.size(), train_features.size(), 
                                10, 10, 2, 1);
    for(int i = 0; i < test_features.size(); i++){
        vector<double> features = test_features[i];
        string tmp = predict(decision_tree, features);
        test_labels.push_back(tmp);
    }

    for(auto s : test_labels) cout << s << endl;
}