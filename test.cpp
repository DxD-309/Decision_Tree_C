#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <limits>
#include <algorithm>
#include <numeric>

using namespace std;

// Struct đại diện một nút trong Decision Tree
struct TreeNode {
    string label; // Nhãn nếu là lá
    int feature;  // Chỉ số đặc trưng để split
    double threshold; // Ngưỡng để chia
    TreeNode* left;
    TreeNode* right;

    TreeNode(string lbl) : label(lbl), feature(-1), threshold(0), left(nullptr), right(nullptr) {}
    TreeNode(int feat, double thresh) : label(""), feature(feat), threshold(thresh), left(nullptr), right(nullptr) {}
};

// Hàm tính độ Gini Impurity
double calculateGini(const vector<string>& labels) {
    map<string, int> count;
    for (const string& label : labels) {
        count[label]++;
    }
    double gini = 1.0;
    int total = labels.size();
    for (const auto& i : count) {
        double prob = (double)i.second / total;
        gini -= prob * prob;
    }
    return gini;
}

// Hàm tính Gini Impurity của một phép chia
double calculateSplitGini(const vector<string>& leftLabels, const vector<string>& rightLabels) {
    int total = leftLabels.size() + rightLabels.size();
    double leftGini = calculateGini(leftLabels);
    double rightGini = calculateGini(rightLabels);

    return (leftLabels.size() * leftGini + rightLabels.size() * rightGini) / total;
}

// Hàm tìm điểm chia tốt nhất
pair<double, double> findBestSplit(const vector<vector<double>>& data, const vector<string>& labels, int feature) {
    set<double> uniqueValues;
    for (const auto& row : data) {
        uniqueValues.insert(row[feature]);
    }

    double bestGini = numeric_limits<double>::max();
    double bestThreshold = -1;

    for (double threshold : uniqueValues) {
        vector<string> leftLabels, rightLabels;
        for (int i = 0; i < data.size(); ++i) {
            if (data[i][feature] <= threshold) {
                leftLabels.push_back(labels[i]);
            } else {
                rightLabels.push_back(labels[i]);
            }
        }
        double gini = calculateSplitGini(leftLabels, rightLabels);
        if (gini < bestGini) {
            bestGini = gini;
            bestThreshold = threshold;
        }
    }
    return {bestGini, bestThreshold};
}

// Hàm xây dựng cây đệ quy
TreeNode* buildTree(const vector<vector<double>>& data, const vector<string>& labels, int depth = 0) {
    // Nếu tất cả nhãn giống nhau, trả về nút lá
    set<string> uniqueLabels(labels.begin(), labels.end());
    if (uniqueLabels.size() == 1) {
        return new TreeNode(*uniqueLabels.begin());
    }

    // Tìm điểm chia tốt nhất
    double bestGini = numeric_limits<double>::max();
    int bestFeature = -1;
    double bestThreshold = -1;
    for (int feature = 0; feature < data[0].size(); ++feature) {
        auto best_split = findBestSplit(data, labels, feature);
        double gini = best_split.first;
        int threshold = best_split.second;
        if (gini < bestGini) {
            bestGini = gini;
            bestFeature = feature;
            bestThreshold = threshold;
        }
    }

    if (bestFeature == -1) {
        // Trả về nhãn phổ biến nhất nếu không thể chia tiếp
        map<string, int> count;
        for (const string& label : labels) {
            count[label]++;
        }
        string mostCommonLabel = max_element(count.begin(), count.end(),
                                             [](const auto& a, const auto& b) { return a.second < b.second; })
                                     ->first;
        return new TreeNode(mostCommonLabel);
    }

    // Chia dữ liệu
    vector<vector<double>> leftData, rightData;
    vector<string> leftLabels, rightLabels;
    for (int i = 0; i < data.size(); ++i) {
        if (data[i][bestFeature] <= bestThreshold) {
            leftData.push_back(data[i]);
            leftLabels.push_back(labels[i]);
        } else {
            rightData.push_back(data[i]);
            rightLabels.push_back(labels[i]);
        }
    }

    // Tạo nút và đệ quy xây dựng cây con
    TreeNode* node = new TreeNode(bestFeature, bestThreshold);
    node->left = buildTree(leftData, leftLabels, depth + 1);
    node->right = buildTree(rightData, rightLabels, depth + 1);

    return node;
}

// Hàm dự đoán dựa trên cây
string predict(TreeNode* root, const vector<double>& sample) {
    if (!root->label.empty()) {
        return root->label;
    }
    if (sample[root->feature] <= root->threshold) {
        return predict(root->left, sample);
    } else {
        return predict(root->right, sample);
    }
}

// Hàm chính
int main() {
    vector<vector<double>> data = {
        {4, 1, 4, 1},
        {5, 3, 5, 3},
        {3, 4, 4, 3},
        {1, 4, 4, 1},
        {1, 5, 5, 1},
        {3, 3, 3, 3},
        {2, 5, 5, 2}
    };
    vector<string> labels = {"L", "B", "L", "R", "R", "B", "B"};

    TreeNode* tree = buildTree(data, labels);

    vector<double> testSample = {3, 4, 4, 2};
    cout << "Prediction: " << predict(tree, testSample) << endl;

    return 0;
}
