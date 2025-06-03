#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

using namespace std;

// Constant for the class label index
const int CLASS_LABEL_INDEX = 0;

// Function to calculate the Euclidean distance between two data points
double euclidean_distance(const vector<double>& p1, const vector<double>& p2) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        sum_sq += pow(p1[i] - p2[i], 2);
    }
    return sqrt(sum_sq);
}

// Function for Leave-One-Out Cross-Validation with Nearest Neighbor
double leave_one_out_cross_validation(const vector<vector<double>>& data, const vector<int>& current_features, int feature_to_add = -1) {
    int num_correctly_classified = 0;
    int num_samples = data.size();
    vector<int> evaluation_features = current_features;
    if (feature_to_add != -1) {
        evaluation_features.push_back(feature_to_add);
    }

    for (int i = 0; i < num_samples; ++i) {
        vector<double> object_to_classify;
        for (int feature_index : evaluation_features) {
            object_to_classify.push_back(data[i][feature_index]);
        }
        int label_object_to_classify = static_cast<int>(data[i][CLASS_LABEL_INDEX]);

        double nearest_neighbor_distance = numeric_limits<double>::max();
        int nearest_neighbor_label = -1;

        for (int k = 0; k < num_samples; ++k) {
            if (i == k) continue;

            vector<double> training_point;
            for (int feature_index : evaluation_features) {
                training_point.push_back(data[k][feature_index]);
            }
            int training_label = static_cast<int>(data[k][CLASS_LABEL_INDEX]);

            double distance = euclidean_distance(object_to_classify, training_point);

            if (distance < nearest_neighbor_distance) {
                nearest_neighbor_distance = distance;
                nearest_neighbor_label = training_label;
            }
        }

        if (label_object_to_classify == nearest_neighbor_label) {
            num_correctly_classified++;
        }
    }

    return static_cast<double>(num_correctly_classified) / num_samples;
}

// Function to check if a feature is already in the set
bool is_feature_present(const vector<int>& feature_set, int feature_index) {
    return find(feature_set.begin(), feature_set.end(), feature_index) != feature_set.end();
}

// Thread pool helper function
void parallel_for(int start, int end, int max_threads, const function<void(int)>& task) {
    mutex mtx;
    condition_variable cv;
    queue<int> tasks;
    atomic<int> active_threads(0); // Initialize atomic variable correctly

    // Populate the task queue
    for (int i = start; i < end; ++i) {
        tasks.push(i);
    }

    auto worker = [&]() {
        while (true) {
            int task_index;
            {
                unique_lock<mutex> lock(mtx);
                if (tasks.empty()) break;
                task_index = tasks.front();
                tasks.pop();
                active_threads.fetch_add(1, memory_order_relaxed); // Increment atomically
            }

            // Execute the task
            task(task_index);

            {
                unique_lock<mutex> lock(mtx);
                active_threads.fetch_sub(1, memory_order_relaxed); // Decrement atomically
                cv.notify_all();
            }
        }
    };

    vector<thread> threads;
    for (int i = 0; i < max_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Forward Selection
vector<int> forward_selection(const vector<vector<double>>& data) {
    int num_features = data[0].size() - 1; // Exclude the class label
    vector<int> current_set_of_features;
    vector<int> best_set_of_features;
    double best_so_far_accuracy = 0.0;

    for (int i = 1; i <= num_features; ++i) {
        cout << "On level " << i << " of the search tree" << endl;
        int feature_to_add_at_this_level = -1;
        double best_accuracy_at_this_level = 0.0;

        mutex mtx; // Mutex to protect shared variables

        parallel_for(1, num_features + 1, 4, [&](int k) {
            if (!is_feature_present(current_set_of_features, k)) {
                vector<int> temp_features = current_set_of_features;
                temp_features.push_back(k);

                double accuracy = leave_one_out_cross_validation(data, temp_features);

                lock_guard<mutex> lock(mtx); // Protect shared variables
                cout << "--Considering adding the feature at index " << k << endl;
                cout << "--Accuracy with feature " << k << ": " << fixed << setprecision(4) << accuracy << endl;

                if (accuracy > best_accuracy_at_this_level) {
                    best_accuracy_at_this_level = accuracy;
                    feature_to_add_at_this_level = k;
                }
            }
        });

        if (feature_to_add_at_this_level != -1) {
            current_set_of_features.push_back(feature_to_add_at_this_level);
            cout << "On level " << i << ", added feature " << feature_to_add_at_this_level
                 << " (Accuracy: " << fixed << setprecision(4) << best_accuracy_at_this_level << ")" << endl;

            if (best_accuracy_at_this_level > best_so_far_accuracy) {
                best_so_far_accuracy = best_accuracy_at_this_level;
                best_set_of_features = current_set_of_features;
            }
        }
    }

    cout << "\nBest set of selected features: ";
    for (size_t i = 0; i < best_set_of_features.size(); ++i) {
        cout << best_set_of_features[i] << (i == best_set_of_features.size() - 1 ? "" : ", ");
    }
    cout << endl;
    cout << "Best accuracy achieved: " << fixed << setprecision(4) << best_so_far_accuracy << endl;

    return best_set_of_features;
}

// Backward Elimination
vector<int> backward_elimination(const vector<vector<double>>& data) {
    int num_features = data[0].size() - 1;
    vector<int> current_set_of_features(num_features);
    iota(current_set_of_features.begin(), current_set_of_features.end(), 1); // Start with all features (excluding class label)
    vector<int> best_set_of_features = current_set_of_features;
    double best_so_far_accuracy = leave_one_out_cross_validation(data, current_set_of_features);

    for (int i = num_features; i > 0; --i) {
        cout << "On level " << i << " of the search tree" << endl;
        int feature_to_remove_at_this_level = -1;
        double best_accuracy_at_this_level = 0.0;

        mutex mtx; // Mutex to protect shared variables

        parallel_for(0, current_set_of_features.size(), 4, [&](int k) {
            vector<int> temp_features = current_set_of_features;
            temp_features.erase(temp_features.begin() + k); // Remove feature k

            double accuracy = leave_one_out_cross_validation(data, temp_features);

            lock_guard<mutex> lock(mtx); // Protect shared variables
            cout << "--Considering removing the feature at index " << current_set_of_features[k] << endl;
            cout << "--Accuracy without feature " << current_set_of_features[k] << ": " << fixed << setprecision(4) << accuracy << endl;

            if (accuracy > best_accuracy_at_this_level) {
                best_accuracy_at_this_level = accuracy;
                feature_to_remove_at_this_level = k;
            }
        });

        if (feature_to_remove_at_this_level != -1) {
            cout << "On level " << i << ", removed feature " << current_set_of_features[feature_to_remove_at_this_level]
                 << " (Accuracy: " << fixed << setprecision(4) << best_accuracy_at_this_level << ")" << endl;
            current_set_of_features.erase(current_set_of_features.begin() + feature_to_remove_at_this_level);

            if (best_accuracy_at_this_level > best_so_far_accuracy) {
                best_so_far_accuracy = best_accuracy_at_this_level;
                best_set_of_features = current_set_of_features;
            }
        }
    }

    cout << "\nBest set of selected features: ";
    for (size_t i = 0; i < best_set_of_features.size(); ++i) {
        cout << best_set_of_features[i] << (i == best_set_of_features.size() - 1 ? "" : ", ");
    }
    cout << endl;
    cout << "Best accuracy achieved: " << fixed << setprecision(4) << best_so_far_accuracy << endl;

    return best_set_of_features;
}

// Function to read data from a file
// vector<vector<double>> read_data_from_file(const string& filename) {
//     vector<vector<double>> data;
//     ifstream file(filename);

//     if (!file.is_open()) {
//         cerr << "Error: Could not open file " << filename << endl;
//         exit(1);
//     }

//     string line;
//     while (getline(file, line)) {
//         vector<double> row;
//         stringstream ss(line);
//         double value;
//         while (ss >> value) {
//             row.push_back(value);
//         }
//         if (!row.empty()) {
//             data.push_back(row);
//         }
//     }

//     file.close();
//     return data;
// }


vector<vector<double>> read_data_from_file(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        // Sticking to your program's style of exiting, though returning empty or throwing is often preferred.
        exit(1); 
    }

    string line;
    int line_number = 0; // For more informative error messages
    while (getline(file, line)) {
        line_number++;
        
        // Preprocess the line: replace all commas with spaces
        string processed_line = line;
        for (char &c : processed_line) {
            if (c == ',') {
                c = ' ';
            }
        }

        vector<double> row;
        stringstream ss(processed_line); // Use the line with commas replaced by spaces
        string cell_str; // Read each potential number as a string
        
        // Now, extract whitespace-separated "words" (which should be numbers)
        while (ss >> cell_str) {
            try {
                // Attempt to convert the cell string to a double
                row.push_back(stod(cell_str)); 
            } catch (const std::invalid_argument& ia) {
                cerr << "Warning (line " << line_number << "): Invalid data format. Could not convert '" << cell_str << "' to double. Original line: '" << line << "'. Skipping cell." << endl;
            } catch (const std::out_of_range& oor) {
                cerr << "Warning (line " << line_number << "): Data out of range for double: '" << cell_str << "'. Original line: '" << line << "'. Skipping cell." << endl;
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}
// Function to normalize the features (excluding the class label)
void normalize_data(vector<vector<double>>& data) {
    if (data.empty() || data[0].size() <= 1) {
        return;
    }

    int num_features = data[0].size() - 1;
    int num_samples = data.size();

    vector<double> means(num_features, 0.0);
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            means[j] += data[i][j + 1];
        }
        means[j] /= num_samples;
    }

    vector<double> stddevs(num_features, 0.0);
    for (int j = 0; j < num_features; ++j) {
        for (int i = 0; i < num_samples; ++i) {
            stddevs[j] += pow(data[i][j + 1] - means[j], 2);
        }
        stddevs[j] = sqrt(stddevs[j] / num_samples);
        if (stddevs[j] == 0.0) {
            stddevs[j] = 1.0;
        }
    }

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            data[i][j + 1] = (data[i][j + 1] - means[j]) / stddevs[j];
        }
    }
}

int main() {
    string filename;
    cout << "Enter the filename: ";
    cin >> filename;

    vector<vector<double>> my_data = read_data_from_file(filename);

    if (my_data.empty()) {
        cerr << "Error: No data read. Exiting." << endl;
        return 1;
    }

    normalize_data(my_data);
   
    

    int choice;
    cout << "Choose search method:\n1. Forward Selection\n2. Backward Elimination\n";
    cin >> choice;

    vector<int> selected_features;
    if (choice == 1) {
        selected_features = forward_selection(my_data);
    } else if (choice == 2) {
        selected_features = backward_elimination(my_data);
    } else {
        cerr << "Invalid choice. Exiting." << endl;
        return 1;
    }

    return 0;
}