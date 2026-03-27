#include <cstdio>
#include <iterator>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <stereo.h>
#include <opencv2/opencv.hpp>

stereoCam::stereoCam(){
}

void stereoCam::initStereo(Mat imageL, Mat imageR){
    sgbm = StereoSGBM::create(0, 16, 3);

    FileStorage fs("intrinsics.yml", FileStorage::READ); 
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;

    fs.open("extrinsics.yml", FileStorage::READ);
    fs["R"] >> R;
    fs["T"] >> T;

    img_size = imageL.size();
    stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);
    initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    numberOfDisparities = ((img_size.width/8) + 15) & -16;

    sgbm->setPreFilterCap(63);
    sgbmWinSize = 3;
    cn = imageL.channels();

    sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
    sgbm->setP1(32 * cn * sgbmWinSize * sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    cout << "P1 size: " << P1.size() << "\n";
    f = P1.at<double>(0,0);
    B = norm(T);
}


vector<double> stereoCam::get_depths(vector<Point> coordinates, Mat imageL, Mat imageR){
    vector<double> depths;
    Mat disp = getDisparity(imageL, imageR);

    for(int i = 0; i < coordinates.size(); i++){
	int x = coordinates[i].x;
	int y = coordinates[i].y;
	float d = disp.at<short>(y, x) / 16.0f;
	if(d > 0){
	    double depth = (B * f) / d;
	    depths.push_back(depth);
	}
    }
    // Mat dst;
    // undistort(imageL, dst, M1, D1);
    // imshow("left", dst);

    return depths;
}

Mat stereoCam::getDisparity(Mat imageL, Mat imageR){


    // NOTE: Rectify Image code
    // Commented cause calibration is bad

    // remap(imageL, img1r, map11, map12, INTER_LINEAR);
    // remap(imageR, img2r, map21, map22, INTER_LINEAR);
    //
    // imageL = img1r;
    // imageR = img2r;

    Mat disp, disp8;

    float disparity_multiplier = 1.0f;

    int64 t = getTickCount();
    sgbm->compute(imageL, imageR, disp);
    if(disp.type() == CV_16S){
	disparity_multiplier = 16.0f;
    }

    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    t = getTickCount() - t;
    cout << "Time elapsed: " << t * 1000 / getTickFrequency() << "\n";

    if(display){
	std::ostringstream oss;
	oss << "disparity sgbm";
	oss << " blocksize:" << sgbmWinSize;
	oss << " max-disparity:" << numberOfDisparities;
	std::string disp_name = oss.str();

	namedWindow("left", cv::WINDOW_NORMAL);
	imshow("left", imageL);
	namedWindow("right", cv::WINDOW_NORMAL);
	imshow("right", imageR);
	namedWindow(disp_name, cv::WINDOW_NORMAL);
	imshow(disp_name, disp8);
    }
    return disp8;
}

void stereoCam::calibrate(Size inputBoardSize, float squareSize, float markerSize, aruco::PredefinedDictionaryType arucoDict, 
			   std::string arucoDictFile, bool displayCorners, bool useCalibrated, bool showRectified){

    vector<Mat> images = take_images();

    vector<vector<Point2f>> imagePoints[2];
    vector<vector<Point3f>> objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)images.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<Mat> goodImages;

    Size boardSizeInnerCorners, boardSizeUnits;

    boardSizeUnits = inputBoardSize;
    boardSizeInnerCorners.width = inputBoardSize.width - 1;
    boardSizeInnerCorners.height = inputBoardSize.height - 1;

    aruco::Dictionary dictionary;
    if(arucoDictFile == "None"){
	dictionary = aruco::getPredefinedDictionary(arucoDict);
    }else{
	FileStorage dict_file(arucoDictFile, FileStorage::Mode::READ);
	FileNode fn(dict_file.root());
	dictionary.readDictionary(fn);
    }
    aruco::CharucoBoard ch_board(boardSizeUnits, squareSize, markerSize, dictionary);
    aruco::CharucoDetector ch_detector(ch_board);
    vector<int> markerIds;

    for(int i = j = 0; i < nimages; i++){
	for(k = 0; k < 2; k++){
	    Mat cimg = images[i*2+k];
	    Mat img;
	    cvtColor(cimg, img, COLOR_BGR2GRAY);
	    if(img.empty()){
		break;
	    }
	    if(imageSize == Size()){
		imageSize = img.size();
	    }else if(img.size() != imageSize){
		cout << "The image has a different size\n";
		break;
	    }
	    bool found = false;
	    vector<Point2f>& corners = imagePoints[k][j];
	    
	    ch_detector.detectBoard(img, corners, markerIds);
	    found = corners.size() == (size_t) (boardSizeInnerCorners.height*boardSizeInnerCorners.width);
	    
	    if(displayCorners){
		Mat cimg1;
		drawChessboardCorners(cimg, boardSizeInnerCorners, corners, found);
		double sf = 640./MAX(img.rows, img.cols);
		resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
		imshow("corners", cimg1);
		char c = (char)waitKey(500);
		if(c == 27 || c == 'q' || c == 'Q'){
		    exit(-1);
		}
	    }else{
		putchar('.');
	    }
	}
	if(k == 2) {
	    goodImages.push_back(images[i*2]);
	    goodImages.push_back(images[i*2+1]);
	    j++;
	}
    }
    cout << j << " pairs have been successfully detected. \n";
    nimages = j;
    if(nimages < 2){
	cout << "ERROR: Too Little Images\n";
	return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for(i = 0; i < nimages; i++){
	for(j = 0; j < boardSizeInnerCorners.height; j++){
	    for(k = 0; k < boardSizeInnerCorners.width; k++){
		objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	    }
	}
    }

    cout << "Running Stereo Calibration ... \n";

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], 
				 cameraMatrix[0], distCoeffs[0],
				 cameraMatrix[1], distCoeffs[1],
				 imageSize, R, T, E, F,
				 CALIB_FIX_ASPECT_RATIO +
				 CALIB_ZERO_TANGENT_DIST +
				 CALIB_USE_INTRINSIC_GUESS + 
				 CALIB_SAME_FOCAL_LENGTH +
				 CALIB_RATIONAL_MODEL +
				 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
				 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error = " << rms << endl;

    // CALIBRATION QUALITY
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(i = 0; i < nimages; i++){
	int npt = (int)imagePoints[0][i].size();
	Mat imgpt[2];
	for(k = 0; k < 2; k++){
	    imgpt[k] = Mat(imagePoints[k][i]);
	    undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
	    computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
	}
	for(j = 0; j < npt; j++){
	    double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
			   fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);	
	    err += errij;
	}
	npoints += npt;
    }
    cout << "average epipolar err = " << err/npoints << endl;

    // Save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if(fs.isOpened()){
	fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
	fs.release();
    }else{
	cout << "ERROR: could not save intrinsic parameters\n";
    }

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0], 
		  cameraMatrix[1], distCoeffs[1],
		  imageSize, R, T, R1, R2, P1, P2, Q,
		  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if(fs.isOpened()){
	fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
	fs.release();
    }else{
	cout << "ERROR: could not save extrinsic parameters\n";
    }

    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    isVerticalStereo = false;

    if(!showRectified){
	return;
    }
    
    Mat rmap[2][2];
    if(useCalibrated){

    }else{
	vector<Point2f> allimgpt[2];
	for(k = 0; k < 2; k++){
	    for(i = 0; i < nimages; i++){
		copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
	    }
	}
	F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
	Mat H1, H2;
	stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

	R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
	R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
	P1 = cameraMatrix[0];
	P2 = cameraMatrix[1];
    }

    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if(!isVerticalStereo){
	sf = 600./MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h, w*2, CV_8UC3);
    }else{
	sf = 300./MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h*2, w, CV_8UC3);
    }

    for(i = 0; i < nimages; i++){
	for(k = 0; k < 2; k++){
	    Mat img, rimg, cimg;
	    cimg = goodImages[i*2+k];
	    cvtColor(cimg, img, COLOR_BGR2GRAY);
	    remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
	    Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
	    resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
	    if(useCalibrated){
		Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
			    cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
		rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
	    }
	}

	if(!isVerticalStereo){
	    for(j = 0; j < canvas.rows; j += 16){
		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
	    }
	}else{
	    for(j = 0; j < canvas.cols; j += 16){
		line(canvas,Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
	    }
	}
	imshow("rectified", canvas);
	char c = (char)waitKey();
	if(c == 27 || c == 'q' || c == 'Q'){
	    break;
	}
    }
}

vector<Mat> stereoCam::take_images(){
    VideoCapture capL(0);
    VideoCapture capR(2);
    vector<Mat> images;
    Mat frameL, frameR, combinedView;
    
    while(true){
	capL >> frameL;
	capR >> frameR;
	
	hconcat(frameL, frameR, combinedView);
	namedWindow("Stereo View", WINDOW_NORMAL);
        imshow("Stereo View", combinedView);

	int key = waitKey(30);
	if(key == 27){
	    cout << "Exited Image Taking\n";
	    break;
	}
	if(key == 32){
	    cout << "Pushed Back Pair\n"; 
	    images.push_back(frameL);
	    images.push_back(frameR);
	}

    }

    capR.release();
    capL.release();
    cv::destroyAllWindows();

    return images;
}
