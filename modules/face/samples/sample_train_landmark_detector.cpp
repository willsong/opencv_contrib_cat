#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | give the following arguments in following format }"
        "{ annotations a  |.     | (required) path to annotations txt file [example - /data/annotations.txt] }"
        "{ config c       |      | (required) path to configuration xml file containing parameters for training.[ example - /data/config.xml] }"
        "{ model m        |      | (required) path to configuration xml file containing parameters for training.[ example - /data/model.dat] }"
        "{ width w        |  460 | The width which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ height h       |  460 | The height which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ face_cascade f |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    //Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string directory(parser.get<string>("annotations"));
    //default initialisation
    Size scale(460,460);
    scale = Size(parser.get<int>("width"),parser.get<int>("height"));
    if (directory.empty()){
        parser.printMessage();
        cerr << "The name of the directory from which annotations have to be found is empty" << endl;
        return -1;
    }
    string configfile_name(parser.get<string>("config"));
    if (configfile_name.empty()){
        parser.printMessage();
        cerr << "No configuration file name found which contains the parameters for training" << endl;
        return -1;
    }
    string modelfile_name(parser.get<string>("model"));
    if (modelfile_name.empty()){
        parser.printMessage();
        cerr << "No name  for the model_file found in which the trained model has to be saved" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "The name of the cascade classifier to be loaded to detect faces is not found" << endl;
        return -1;
    }
    //create a vector to store names of files in which annotations
    //and image names are found
    /*The format of the file containing annotations should be of following format
        /data/abc/abc.jpg
        123.45,345.65
        321.67,543.89
        The above format is similar to HELEN dataset which is used for training model
     */
    vector<String> filenames;
    //reading the files from the given directory
    glob(directory + "*.txt",filenames);
    cout << "willsong debug 1: " << directory << ", " << filenames.size() << endl;
    //create a pointer to call the base class
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    cout << "willsong debug 2: " << cascade_name << endl;
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
    cout << "willsong debug 3: " << configfile_name << endl;
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);
    cout << "willsong debug 4" << endl;
    //create a vector to store image names
    vector<String> imagenames;
    //create object to get landmarks
    vector< vector<Point2f> > trainlandmarks,Trainlandmarks;
    //gets landmarks and corresponding image names in both the vectors
    //vector to store images
    vector<Mat> trainimages;
    loadTrainingData(filenames,trainlandmarks,imagenames);
    cout << "willsong debug 5: " << imagenames.size() << endl;
    for(unsigned long i=0;i<imagenames.size();i++){
        cout << "willsong debug 6: " << imagenames[i] << endl;
        //string imgname = imagenames[i].substr(0, imagenames[i].size()-1);
        string img = directory + imagenames[i];
        cout << "willsong debug 7: " << img << endl;
        Mat src = imread(img);
        if(src.empty()){
            cerr<<string("Image "+img+" not found\n.")<<endl;
            continue;
        }
        trainimages.push_back(src);
        Trainlandmarks.push_back(trainlandmarks[i]);
    }
    cout<<"Got data"<<endl;
    cout << "willsong debug8: " << trainimages.size() << endl;
    facemark->training(trainimages,Trainlandmarks,configfile_name,scale,modelfile_name);
    cout<<"Training complete"<<endl;
    return 0;
}
