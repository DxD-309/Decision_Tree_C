#include <bits/stdc++.h>
using namespace std;

string train_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/train.csv";
string test_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/test.csv";
string return_file = "C:/Users/Admin/Desktop/VSCODE/Decision Tree/predict_5.txt";

void extract_csv(vector<vector<double>>& features, vector<string>& labels, string filename);
void return_text(vector<string>& predict, string return_file);
struct Node;
double gini_score(vector<string>& labels);
double calc_total_gini(vector<string>& left_gini, vector<string>& right_gini);
pair<double, double> find_best_split(vector<vector<double>>& features, vector<string>& labels, int feature);
Node* build(vector<vector<double>>& features, vector<string>& labels, int depth, int leaf, int sample_split, int sample_leaf,
            int max_depth, int max_leaf, int min_sample_split, int min_sample_leaf);
string predict(Node* root, vector<double>& samples);
vector<vector<int>> confusion_matrix(vector<string>& pred, vector<string>& label);
double precision_score(vector<vector<int>>& confusion_matrix, int i);
double recall_score(vector<vector<int>>& confusion_matrix, int i);
double f1_score(vector<vector<int>>& confusion_matrix, int i);
double f1_macro(vector<vector<int>>& confusion_matrix);
double f1_micro(vector<vector<int>>& confusion_matrix);
double cross_validation(vector<vector<double>>& features, vector<string>& labels, int k, int f1_type, int max_depth, 
                        int max_leaf, int min_sample_split, int min_sample_leaf);
vector<pair<string, double>> grid_search_cv(vector<vector<double>>& features, vector<string>& labels, int k,
        vector<int>& max_depth_range, vector<int>& max_leaf_range, vector<int>& min_sample_split_range, 
        vector<int>& min_sample_leaf_range);

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
    file.close();
}

void return_text(vector<string>& predict, string return_file){
    ofstream file(return_file);
    if(!file.is_open()){
        cout << "Error!";
        return;
    }
    for(unsigned int i = 0; i < predict.size(); i++){
        file << predict[i];
        file << "\n";
    }
    file.close();
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
    for(unsigned int i = 0; i < features.size(); i++){
        featureSet.insert(features[i][feature]);
    }
    double best_threshold = -1;
    double best_gini_score = 100000000.0;
    for(double threshold : featureSet){
        vector<string> left_gini;
        vector<string> right_gini;
        for(unsigned int i = 0; i < features.size(); i++){
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
    for(unsigned int feature = 0; feature < features[0].size(); feature++){
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
    for(unsigned int i = 0; i < features.size(); i++){
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

    if(tp + fp == 0.0) return 0.0;
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

    if(tp + fn == 0.0) return 0.0;
    return tp/(tp+fn);
}

double f1_score(vector<vector<int>>& confusion_matrix, int i){
    double precision = precision_score(confusion_matrix, i);
    double recall = recall_score(confusion_matrix, i);
    if(precision + recall == 0.0) return 0.0;
    return (2*precision*recall)/(precision+recall);
}

double f1_macro(vector<vector<int>>& confusion_matrix){
    double F1_L = f1_score(confusion_matrix, 0);
    double F1_R = f1_score(confusion_matrix, 1);
    double F1_B = f1_score(confusion_matrix, 2);
    return (F1_L+F1_R+F1_B)/3.0;
}

double f1_micro(vector<vector<int>>& confusion_matrix){
    double tp_total = (double)(confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]);
    double fp_total = (double)(confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][0]
                            + confusion_matrix[1][2] + confusion_matrix[2][0] + confusion_matrix[2][1]);
    double fn_total = (double)(confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[0][1]
                            + confusion_matrix[2][1] + confusion_matrix[0][2] + confusion_matrix[1][2]);

    return (2*tp_total)/(2*tp_total + fp_total + fn_total);
}

double cross_validation(vector<vector<double>>& features, vector<string>& labels, int fold, string f1_type, int max_depth, 
                        int max_leaf, int min_sample_split, int min_sample_leaf){
    int fold_size = features.size()/fold;
    double f1_score = 0;
    for(int i = 0; i < fold; i++){
        vector<vector<double>> train_feature, test_feature;
        vector<string> train_label, test_label;

        int n = features.size();
        for(int j = 0; j < n; j++){
            if(j/fold_size == i){
                test_feature.push_back(features[j]);
                test_label.push_back(labels[j]);
            }
            else{
                train_feature.push_back(features[j]);
                train_label.push_back(labels[j]);
            }
        }

        Node* decision_tree = build(train_feature, train_label, 1, 1, train_feature.size(), train_feature.size(),
                                    max_depth, max_leaf, min_sample_split, min_sample_leaf);

        vector<string> pred;
        for(unsigned int i = 0; i < test_feature.size(); i++){
            string tmp = predict(decision_tree, test_feature[i]);
            pred.push_back(tmp);
        }

        vector<vector<int>> confusion_matrixx = confusion_matrix(pred, test_label);

        double tmp_f1_score;
        if(f1_type == "macro"){
            tmp_f1_score = f1_macro(confusion_matrixx);
        }
        else{
            tmp_f1_score = f1_micro(confusion_matrixx);
        }
        f1_score += tmp_f1_score;
    }
    return f1_score/(double)fold;
}

vector<pair<string, double>> grid_search_cv(vector<vector<double>>& features, vector<string>& labels, int fold,
        vector<int>& max_depth_range, vector<int>& max_leaf_range, vector<int>& min_sample_split_range, 
        vector<int>& min_sample_leaf_range){
    int best_max_depth = -1;
    int best_max_leaf = -1;
    int best_min_sample_split = -1;
    int best_min_sample_leaf = -1;
    double best_cv_score = -1000000;
    for(unsigned int i = 0; i < max_depth_range.size(); i++){
        for(unsigned int j = 0; j < max_leaf_range.size(); j++){
            for(unsigned int k = 0; k < min_sample_split_range.size(); k++){
                for(unsigned int l = 0; l < min_sample_leaf_range.size(); l++){
                    double tmp_cv_score = cross_validation(features, labels, fold, "micro", max_depth_range[i],
                    max_leaf_range[j], min_sample_split_range[k], min_sample_leaf_range[l]);
                    if(tmp_cv_score > best_cv_score){
                        best_cv_score = tmp_cv_score;
                        best_max_depth = max_depth_range[i];
                        best_max_leaf = max_leaf_range[j];
                        best_min_sample_split = min_sample_split_range[k];
                        best_min_sample_leaf = min_sample_leaf_range[l];
                    }
                }
            }
        }
    }
    vector<pair<string, double>> ans;
    ans.push_back({"Best F1 Score: ", best_cv_score});
    ans.push_back({"Best Max Depth: ", best_max_depth});
    ans.push_back({"Best Max Leaf: ", best_max_leaf});
    ans.push_back({"Best Min Sample Split: ", best_min_sample_split});
    ans.push_back({"Best Min Sample Leaf: ", best_min_sample_leaf});
    return ans;
}

int main(){
    vector<vector<double>> train_features;
    vector<string> train_labels;
    vector<vector<double>> test_features;
    vector<string> test_labels;
    extract_csv(train_features, train_labels, train_file);
    extract_csv(test_features, test_labels, test_file);

    vector<int> max_depth_range;
    vector<int> max_leaf_range;
    vector<int> min_sample_split_range;
    vector<int> min_sample_leaf_range;
        
    for(int i = 5; i <= 10; i++) max_depth_range.push_back(i);
    for(int i = 1; i <= 10; i++){
        max_leaf_range.push_back(i);
    }
    for(int i = 1; i <= 10; i++){
        min_sample_split_range.push_back(i);
        min_sample_leaf_range.push_back(i);
    }

    // vector<pair<string, double>> grid_search = grid_search_cv(train_features, train_labels, 10, max_depth_range, 
    //                         max_leaf_range, min_sample_split_range, min_sample_leaf_range);

    // for(auto p : grid_search){
    //     cout << p.first << p.second << endl;
    // }

    // Best F1 Score: 0.803774 
    // Best Max Depth: 9       
    // Best Max Leaf: 2        
    // Best Min Sample Split: 1
    // Best Min Sample Leaf: 5

    // Best F1 Score: 0.79434
    // Best Max Depth: 10
    // Best Max Leaf: 8
    // Best Min Sample Split: 1
    // Best Min Sample Leaf: 1


    Node* decision_tree = build(train_features, train_labels, 1, 1, train_features.size(), train_features.size(), 
                                9, 2, 1, 5);
    for(unsigned int i = 0; i < test_features.size(); i++){
        vector<double> features = test_features[i];
        string tmp = predict(decision_tree, features);
        test_labels.push_back(tmp);
    }

    for(auto s : test_labels) cout << s << endl;
    return_text(test_labels, return_file);
}