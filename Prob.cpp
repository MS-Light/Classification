#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
K = 1

class Model{
public:
    bool saveToFile(string fileName);
    bool loadFromFile(string fileName);
double probabilities[28][28][10][2];
double prob_of_class[10];
};

bool Model::saveToFile(string fileNmae) {
    ofstream saveFile("../data/" + string(fileName) + ".txt");
    if (!saveFile) {
        return false;
    }

    for (int num = 0; num < 10; num++) {
        for (int width = 0; width < DATUM_WIDTH; width++) {
            for (int height = 0; height < DATUM_HEIGHT; height++) {
                for (int binary = 0; binary < 2; binary++) {
                    saveFile << setw(12) << probabilities[width][height][num][binary] << setw(12);
                }
            }
            saveFile << endl;
        }
        saveFile << endl;
    }
    saveFile.close();
    return true;
}

bool Model::loadFromFile(string fileName) {
    double data = 0.0;
    int i = 0;
    int j = 0;
    int classNum = 0;
    int binary = 0;

    ifstream inFile("../data/" + fileName);

    if (!inFile.is_open()) {
        cout << "The file is not existed! Please input the correct name" <<endl;
        exit(0);

    }
    while(inFile >> data) {
        probabilities[i][j][classNum][binary] = data;
        binary++;
        if (binary > 1) {
            j++;
            binary = 0;
        }
        if (j > 27) {
            i++;
            j = 0;
        }
        if (i > 27) {
            classNum++;
            i=0;
            j=0;
        }
    }
    return true;
}

void addModeProbability(Model& model, const vector<int> &labels, const vector<ImageData> &trainingData){
    for (unsigned long pos = 0; pos < trainingData.size(); pos++) {
        ImageData currentImage = trainingData.at((pos));
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                bool currentBoolean = currentImage.Image[i][j];
                model.probabilities[i][j][labels[pos]][currentBoolean]++;
            }
        }
    }
}
void makeProbability(Model& model, const int arr[10]) {
    for (int width = 0; width < IMAGE_SIZE; width++) {
        for (int height = 0; height < IMAGE_SIZE; height++) {
            for (int num = 0; num < CLASS_NUM; num++) {
                for (int binary = 0; binary < BINARY; binary++) {
                    double temp = model.probabilities[width][height][num][binary];
                    model.probabilities[width][height][num][binary] =
                            (K + temp) / (double)(K + K + arr[num]);
                }
            }
        }
    }

}