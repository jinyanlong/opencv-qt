//#include "mainwindow.h"
//#include <QApplication>

//int main(int argc, char *argv[])
//{
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

//    return a.exec();
//}

//#include <QApplication>

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <zbar.h>
//#include <iostream>
//#include <iomanip>
//#include <QDebug>

//using namespace std;
//using namespace cv;
//using namespace zbar;

//int main(int argc, char **argv) {
////    int cam_idx = 0;

////    if (argc == 2) {
////        cam_idx = atoi(argv[1]);
////    }

////    VideoCapture cap(cam_idx);
////    if (!cap.isOpened()) {
////        cerr << "Could not open camera." << endl;
////        exit(EXIT_FAILURE);
////    }
//    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
//    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

//    namedWindow("captured", CV_WINDOW_AUTOSIZE);

//    // Create a zbar reader
//    ImageScanner scanner;

//    // Configure the reader
//    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

//    // Capture an OpenCV frame
//    cv::Mat frame, frame_grayscale;

////        cap >> frame;
//    frame = imread("9.png");
////    frame = imread("2.jpg");
////    frame = imread("3.jpg");


//    // Convert to grayscale
//    cvtColor(frame, frame_grayscale, CV_BGR2GRAY);

//    // Obtain image data
//    int width = frame_grayscale.cols;
//    int height = frame_grayscale.rows;
//    uchar *raw = (uchar *)(frame_grayscale.data);

//    // Wrap image data
//    Image image(width, height, "Y800", raw, width * height);

//    // Scan the image for barcodes
//    //int n = scanner.scan(image);
//    scanner.scan(image);//扫描条码

//    // Extract results
//    int counter = 0;
//    for (Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol) {
//        time_t now;
//        tm *current;
//        now = time(0);
//        current = localtime(&now);

//        // do something useful with results
//        cout    << "[" << current->tm_hour << ":" << current->tm_min << ":" << setw(2) << setfill('0') << current->tm_sec << "] " << counter << " "
//                << "decoded " << symbol->get_type_name()
//                << " symbol \"" << symbol->get_data() << '"'<<endl;

//        std::string rawdata= symbol->get_data();
//        QByteArray byte = QByteArray::fromRawData(rawdata.c_str(), rawdata.size());
////        QString qstr = QString::from(byte.toHex());
//        qDebug()    <<"data:"<<byte.toHex()<< endl;

////        QString qstr = QString::fromAscii(byte.toHex());

//        //cout << "Location: (" << symbol->get_location_x(0) << "," << symbol->get_location_y(0) << ")" << endl;
//        //cout << "Size: " << symbol->get_location_size() << endl;

//        // Draw location of the symbols found
//        if (symbol->get_location_size() == 4) {
//            //rectangle(frame, Rect(symbol->get_location_x(i), symbol->get_location_y(i), 10, 10), Scalar(0, 255, 0));
//            line(frame, Point(symbol->get_location_x(0), symbol->get_location_y(0)), Point(symbol->get_location_x(1), symbol->get_location_y(1)), Scalar(0, 255, 0), 2, 8, 0);
//            line(frame, Point(symbol->get_location_x(1), symbol->get_location_y(1)), Point(symbol->get_location_x(2), symbol->get_location_y(2)), Scalar(0, 255, 0), 2, 8, 0);
//            line(frame, Point(symbol->get_location_x(2), symbol->get_location_y(2)), Point(symbol->get_location_x(3), symbol->get_location_y(3)), Scalar(0, 255, 0), 2, 8, 0);
//            line(frame, Point(symbol->get_location_x(3), symbol->get_location_y(3)), Point(symbol->get_location_x(0), symbol->get_location_y(0)), Scalar(0, 255, 0), 2, 8, 0);
//        }

//        // Get points
//        /*for (Symbol::PointIterator point = symbol.point_begin(); point != symbol.point_end(); ++point) {
//            cout << point << endl;
//        } */
//        counter++;
//    }

//    // Show captured frame, now with overlays!
//    imshow("captured", frame);
//    waitKey(0);
//    // clean up
//    image.set_data(NULL, 0);



//    return 0;
//}






//#include "zbar.h"
//#include "cv.h"
//#include "highgui.h"
//#include <iostream>

//using namespace std;
//using namespace zbar;  //添加zbar名称空间
//using namespace cv;

//int main(int argc,char*argv[])
//{
//    ImageScanner scanner;
//    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
////    Mat image = imread(argv[1]);
//    Mat image = imread("1.jpeg");
//    Mat imageGray;
//    cvtColor(image,imageGray,CV_RGB2GRAY);
//    int width = imageGray.cols;
//    int height = imageGray.rows;
//    uchar *raw = (uchar *)imageGray.data;
//    Image imageZbar(width, height, "Y800", raw, width * height);
//    scanner.scan(imageZbar); //扫描条码
//    Image::SymbolIterator symbol = imageZbar.symbol_begin();
//    if(imageZbar.symbol_begin()==imageZbar.symbol_end())
//    {
//        cout<<"查询条码失败，请检查图片！"<<endl;
//    }
//    for(;symbol != imageZbar.symbol_end();++symbol)
//    {
//        cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;
//        cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;
//    }
//    imshow("Source Image",image);
//    waitKey();
//    imageZbar.set_data(NULL,0);
//    return 0;
//}


//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

//#include <QDebug>

//using namespace cv;
//using namespace std;

///////////////////////////////////////////////
//double angle( Point pt1, Point pt2, Point pt0 ) {// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
//    double dx1 = pt1.x - pt0.x;
//    double dy1 = pt1.y - pt0.y;
//    double dx2 = pt2.x - pt0.x;
//    double dy2 = pt2.y - pt0.y;
//    double ratio;//边长平方的比
//    ratio=(dx1*dx1+dy1*dy1)/(dx2*dx2+dy2*dy2);
//    if (ratio<0.8 || 1.2<ratio) {//根据边长平方的比过小或过大提前淘汰这个四边形，如果淘汰过多，调整此比例数字
////      Log("ratio\n");
//        return 1.0;//根据边长平方的比过小或过大提前淘汰这个四边形
//    }
//    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
//}
//void findSquares( const Mat& gray0, vector<vector<Point> >& squares ) {// returns sequence of squares detected on the gray0. the sequence is stored in the specified memory storage
//    squares.clear();

//    Mat pyr,gray1,timg;

//    // down-scale and upscale the gray0 to filter out the noise
//    pyrDown(gray0, pyr, Size(gray0.cols/2, gray0.rows/2));
//    pyrUp(pyr, timg, gray0.size());
//    vector<vector<Point> > contours;
//    int N = 100;
//    // try several threshold levels
//    for (int l = 0; l < N; l++ ) {

//        // hack: use Canny instead of zero threshold level.
//        // Canny helps to catch squares with gradient shading
////      if (l == 0 ) {//可试试不对l==0做特殊处理是否能在不影响判断正方形的前提下加速判断过程
////          // apply Canny. Take the upper threshold from slider
////          // and set the lower to 0 (which forces edges merging)
////          Canny(timg, gray1, 0, thresh, 5);
////          // dilate canny output to remove potential
////          // holes between edge segments
////          dilate(gray1, gray1, Mat(), Point(-1,-1));
////      } else {
//            // apply threshold if l!=0:
//            //     tgray(x,y) = gray1(x,y) < (l+1)*255/N ? 255 : 0
//            gray1 = timg >= (l+1)*255/N;
////      }

//        // find contours and store them all as a list
//        findContours(gray1, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

//        vector<Point> approx;

//        // test each contour
//        for (size_t i = 0; i < contours.size(); i++ ) {
//            // approximate contour with accuracy proportional
//            // to the contour perimeter//jyl对图像轮廓点进行多边形拟合, jyl表示输出的多边形点集

//            approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);//0.02为将毛边拉直的系数，如果对毛边正方形漏检，可试试调大
//            // square contours should have 4 vertices after approximation
//            // relatively large area (to filter out noisy contours)
//            // and be convex.
//            // Note: absolute value of an area is used because
//            // area may be positive or negative - in accordance with the
//            // contour orientation//jyl isContourConvex 判断一个轮廓是否是凸包
//            if (approx.size() == 4 && isContourConvex(Mat(approx))) {
//                double area;
//                area=fabs(contourArea(Mat(approx)));
//                if (40.0<area && area<30000.0) {
////                if (4000.0<area && area<30000.0) {//当正方形面积在此范围内……，如果有因面积过大或过小漏检正方形问题，调整此范围。
////                  printf("area=%lg\n",area);
//                    double maxCosine = 0.0;

//                    for (int j = 2; j < 5; j++ ) {
//                        // find the maximum cosine of the angle between joint edges
//                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                        if (maxCosine==1.0) break;// //边长比超过设定范围
//                    }

//                    // if cosines of all angles are small
//                    // (all angles are ~90 degree) then write quandrange
//                    // vertices to resultant sequence
//                    if (maxCosine < 0.1 ) {//四个角和直角相比的最大误差，可根据实际情况略作调整，越小越严格
//                        squares.push_back(approx);
//                        qDebug()<<"@@@@@@@@@@";
//                        return;//检测到一个合格的正方形就返回
////                  } else {
////                      Log("Cosine>=0.1\n");
//                    }
//                }
//            }
//        }
//    }
//}
///////////////////////////////////////////////
///// \brief src
/////
//Mat src; Mat src_gray;
//RNG rng(12345);
////Scalar colorful = CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

////获取轮廓的中心点
//Point Center_cal(vector<vector<Point> > contours,int i)
//{
//      int centerx=0,centery=0,n=contours[i].size();
//      //在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，
//      //求所提取四个点的平均坐标（即为小正方形的大致中心）
//      centerx = (contours[i][n/4].x + contours[i][n*2/4].x + contours[i][3*n/4].x + contours[i][n-1].x)/4;
//      centery = (contours[i][n/4].y + contours[i][n*2/4].y + contours[i][3*n/4].y + contours[i][n-1].y)/4;
//      Point point1=Point(centerx,centery);
//      return point1;
//}


//int main( int argc, char** argv[] )
//{
//    src = imread( "12.png", 1 );
////    src = imread( "91.jpeg", 1 );
////    src = imread( "1.jpeg", 1 );

//    Mat src_all=src.clone();
//    //彩色图转灰度图
//    cvtColor( src, src_gray, CV_BGR2GRAY );
//    //对图像进行平滑处理
////    blur( src_gray, src_gray, Size(3,3) );

//    equalizeHist( src_gray, src_gray );//使灰度图象直方图均衡化//jyl增加对比度

//    /////////////////////////////////////////////////////////
////    Mat drawingj = Mat::zeros( src.size(), CV_8UC3 );
////    vector<vector<Point> > contoursj;
////    findSquares(src_gray,contoursj);
////    for(int i=0; i<contoursj.size(); i++)
////        drawContours( drawingj, contoursj, i,  CV_RGB(rng.uniform(100,255),rng.uniform(100,255),rng.uniform(100,255)) , 1,8);
////    imshow("drawingj",drawingj);
//    ////////////////////////////////////////////////////////

//    namedWindow("src_gray");
//    imshow("src_gray",src_gray);
//    Scalar color = Scalar(1,1,255 );
//    Mat threshold_output;
//    vector<vector<Point> > contours,contours2;
//    vector<Vec4i> hierarchy;
//    Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
//    Mat drawing2 = Mat::zeros( src.size(), CV_8UC3 );
//    Mat drawingAllContours = Mat::zeros( src.size(), CV_8UC3 );

//    //指定112阀值进行二值化
//    threshold( src_gray, threshold_output, 112, 255, THRESH_BINARY );

//    namedWindow("Threshold_output");
//    imshow("Threshold_output",threshold_output);

//    /*查找轮廓
//     *  参数说明
//        输入图像image必须为一个2值单通道图像
//        contours参数为检测的轮廓数组，每一个轮廓用一个point类型的vector表示
//        hiararchy参数和轮廓个数相同，每个轮廓contours[ i ]对应4个hierarchy元素hierarchy[ i ][ 0 ] ~hierarchy[ i ][ 3 ]，
//            分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，该值设置为负数。
//        mode表示轮廓的检索模式
//            CV_RETR_EXTERNAL 表示只检测外轮廓
//            CV_RETR_LIST 检测的轮廓不建立等级关系
//            CV_RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
//            CV_RETR_TREE 建立一个等级树结构的轮廓。具体参考contours.c这个demo
//        method为轮廓的近似办法
//            CV_CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
//            CV_CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
//            CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法
//        offset表示代表轮廓点的偏移量，可以设置为任意值。对ROI图像中找出的轮廓，并要在整个图像中进行分析时，这个参数还是很有用的。
//     */

//    findContours( threshold_output, contours, hierarchy,  CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
////      findContours( threshold_output, contours, hierarchy,  CV_RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );

//    int c=0,ic=0,k=0,area=0;

//    //通过黑色定位角作为父轮廓，有两个子轮廓的特点，筛选出三个定位角
//    int parentIdx=-1;
//    for( int i = 0; i< contours.size(); i++ )
//    {
//        //画出所以轮廓图
//        drawContours( drawingAllContours, contours, parentIdx,  CV_RGB(255,255,255) , 1, 8);
//        if (hierarchy[i][2] != -1 && ic==0)//
//        {
//            parentIdx = i;
//            ic++;
//        }
//        else if (hierarchy[i][2] != -1)//子轮廓
//        {
//            ic++;
//        }
//        else if(hierarchy[i][2] == -1)//父轮廓
//        {
//            ic = 0;
//            parentIdx = -1;
//        }

//        //有两个子轮廓
//        if ( ic >= 2)
//        {
//            qDebug()<<"########";
//            //保存找到的三个黑色定位角
//            contours2.push_back(contours[parentIdx]);
//            //画出三个黑色定位角的轮廓
//            drawContours( drawing, contours, parentIdx,  CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)) , 1, 8);
//            ic = 0;
//            parentIdx = -1;

//        }
//    }

//    填充的方式画出三个黑色定位角的轮廓
//    for(int i=0; i<contours2.size(); i++)
//        drawContours( drawing2, contours2, i,  CV_RGB(rng.uniform(100,255),rng.uniform(100,255),rng.uniform(100,255)) , -1, 4, hierarchy[k][2], 0, Point() );

//    //获取三个定位角的中心坐标
//    Point point[3];
//    for(int i=0; i<contours2.size(); i++)
//    {
//        point[i] = Center_cal( contours2, i );
//    }

//    //计算轮廓的面积，计算定位角的面积，从而计算出边长
//    area = contourArea(contours2[1]);
//    int area_side = cvRound (sqrt (double(area)));
//    for(int i=0; i<contours2.size(); i++)
//    {
//        //画出三个定位角的中心连线
//        line(drawing2,point[i%contours2.size()],point[(i+1)%contours2.size()],color,area_side/2,8);
//    }

//    namedWindow("DrawingAllContours");
//    imshow( "DrawingAllContours", drawingAllContours );

//    namedWindow("Drawing2");
//    imshow( "Drawing2", drawing2 );

//    namedWindow("Drawing");
//    imshow( "Drawing", drawing );


//    //接下来要框出这整个二维码
//    Mat gray_all,threshold_output_all;
//    vector<vector<Point> > contours_all;
//    vector<Vec4i> hierarchy_all;
//    cvtColor( drawing2, gray_all, CV_BGR2GRAY );


//    threshold( gray_all, threshold_output_all, 45, 255, THRESH_BINARY );
//    findContours( threshold_output_all, contours_all, hierarchy_all,  RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );//RETR_EXTERNAL表示只寻找最外层轮廓


//    Point2f fourPoint2f[4];
//    //求最小包围矩形
//    RotatedRect rectPoint = minAreaRect(contours_all[0]);

//    //将rectPoint变量中存储的坐标值放到 fourPoint的数组中
//    rectPoint.points(fourPoint2f);


//    for (int i = 0; i < 4; i++)
//    {
//        line(src_all, fourPoint2f[i%4], fourPoint2f[(i + 1)%4]
//            , Scalar(20,21,237), 3);
//    }

//    namedWindow("Src_all");
//    imshow( "Src_all", src_all );

    //框出二维码后，就可以提取出二维码，然后使用解码库zxing，解出码的信息。
    //或者研究二维码的排布规则，自己写解码部分

//    waitKey(0);
//    return(0);
//}


//////////////////////////////二维码矫正///////////////////////////////////////////////
//#include <QApplication>
//#include <zbar.h>
//#include "cv.h"
//#include "highgui.h"
//#include <iostream>

//using namespace std;
//using namespace zbar;  //添加zbar名称空间
//using namespace cv;

//int main()
//{
//    Mat imageSource=imread("11.png",0);
//    Mat image;
//    imageSource.copyTo(image);
//    GaussianBlur(image,image,Size(3,3),0);  //滤波
//    threshold(image,image,100,255,CV_THRESH_BINARY);  //二值化
//    imshow("二值化",image);
//    Mat element=getStructuringElement(2,Size(7,7));  //膨胀腐蚀核
//    //morphologyEx(image,image,MORPH_OPEN,element);
//    for(int i=0;i<10;i++)
//    {
//        erode(image,image,element);
//        i++;
//    }
//    imshow("腐蚀s",image);
//    Mat image1;
//    erode(image,image1,element);
//    image1=image-image1;
//    imshow("边界",image1);
//    //寻找直线 边界定位也可以用findContours实现
//    vector<Vec2f>lines;
//    HoughLines(image1,lines,1,CV_PI/150,250,0,0);
////    标准霍夫变换:
////    参数1：输入单通道的二值图像；
////    参数2：经过函数HoughLines储存了霍夫变换检测到直线的输出矢量；即需要提取定义一个矢量结构lines用于存放：vector<Vec2f> lines; /<Vec2f>----Vec<float，2>。
////    参数3：double类型的rho,以像素为单位的距离精度。
////    参数4：以弧度表示的累加器的角度分辨率。
////    参数5：阈值累加器阈值参数。 即识别某部分为图中的直线时，它在累加平面中必须达到的值，大于此阈值的线段才可以被检测通过返回到结果中。
////    参数6：double类型的srn,有默认值0
////    参数7：double类型的stn,有默认值0
//    Mat DrawLine=Mat::zeros(image1.size(),CV_8UC1);
//    for(int i=0;i<lines.size();i++)
//    {
//        float rho=lines[i][0];
//        float theta=lines[i][1];
//        Point pt1,pt2;
//        double a=cos(theta),b=sin(theta);
//        double x0=a*rho,y0=b*rho;
//        pt1.x=cvRound(x0+1000*(-b));//对一个double型的数进行四舍五入，并返回一个整型数
//        pt1.y=cvRound(y0+1000*a);
//        pt2.x=cvRound(x0-1000*(-b));
//        pt2.y=cvRound(y0-1000*a);
//        line(DrawLine,pt1,pt2,Scalar(255),1,CV_AA);
//    }
//    imshow("直线",DrawLine);
//    Point2f P1[4];
//    Point2f P2[4];
//    vector<Point2f>corners;
//    goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
//    for(int i=0;i<corners.size();i++)
//    {
//        circle(DrawLine,corners[i],3,Scalar(255),3);
//        P1[i]=corners[i];
//    }
//    imshow("交点",DrawLine);
//    int width=P1[1].x-P1[0].x;
//    int hight=P1[2].y-P1[0].y;
//    P2[0]=P1[0];
//    P2[1]=Point2f(P2[0].x+width,P2[0].y);
//    P2[2]=Point2f(P2[0].x,P2[1].y+hight);
//    P2[3]=Point2f(P2[1].x,P2[2].y);
//    Mat elementTransf;
//    elementTransf=  getAffineTransform(P1,P2);
//    warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
//    仿射变换
//    . src: 输入图像
//    . dst: 输出图像，尺寸由dsize指定，图像类型与原图像一致
//    . M: 2X3的变换矩阵
//    . dsize: 指定图像输出尺寸
//    . flags: 插值算法标识符，有默认值INTER_LINEAR，如果插值算法为WARP_INVERSE_MAP, warpAffine
//           函数使用如下矩阵进行图像转换
//    imshow("校正",imageSource);
//    //Zbar二维码识别
//    ImageScanner scanner;
//    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
//    int width1 = imageSource.cols;
//    int height1 = imageSource.rows;
//    uchar *raw = (uchar *)imageSource.data;
//    Image imageZbar(width1, height1, "Y800", raw, width * height1);
//    scanner.scan(imageZbar); //扫描条码
//    Image::SymbolIterator symbol = imageZbar.symbol_begin();
//    if(imageZbar.symbol_begin()==imageZbar.symbol_end())
//    {
//        cout<<"查询条码失败，请检查图片！"<<endl;
//    }
//    for(;symbol != imageZbar.symbol_end();++symbol)
//    {
//        cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;
//        cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;
//    }
//    namedWindow("Source Window",0);
//    imshow("Source Window",imageSource);
//    waitKey();
//    imageZbar.set_data(NULL,0);
//    return 0;
//}


//////////////////////////////查找正方形///////////////////////////////////////////////
#include <QApplication>
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;


int thresh = 50;
IplImage* img =NULL;
IplImage* img0 = NULL;
CvMemStorage* storage =NULL;
const char * wndname = "正方形检测 demo";

//angle函数用来返回（两个向量之间找到角度的余弦值）
double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
 double dx1 = pt1->x - pt0->x;
 double dy1 = pt1->y - pt0->y;
 double dx2 = pt2->x - pt0->x;
 double dy2 = pt2->y - pt0->y;
 return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// 返回图像中找到的所有轮廓序列，并且序列存储在内存存储器中
CvSeq* findSquares4( IplImage* img, CvMemStorage* storage )
{
 CvSeq* contours;
 int i, c, l, N = 11;
 CvSize sz = cvSize( img->width & -2, img->height & -2 );

 IplImage* timg = cvCloneImage( img );
 IplImage* gray = cvCreateImage( sz, 8, 1 );
 IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
 IplImage* tgray;
 CvSeq* result;//动态结构序列CvSeq是所有OpenCv动态数据结构的基础
 double s, t;
 // 创建一个空序列用于存储轮廓角点
 CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
//创建一序列/seq_flags为序列的符号标志/序列头部的大小/Elem_size为元素的大小
 cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height ));//获取感兴趣的区域ROI
 // 过滤噪音
 cvPyrDown( timg, pyr, 7 );
 //下采样原理:把原始图像s*s窗口内的图像变成一个像素，这个像素点的值就是窗口内所有像素的均值：
 cvPyrUp( pyr, timg, 7 );
// 上采样原理：图像放大几乎都是采用内插值方法，即在原有图像像素的基础上在像素点之间采用合适的插值算法插入新的元素。
 tgray = cvCreateImage( sz, 8, 1 );

 // 红绿蓝3色分别尝试提取
 for( c = 0; c < 3; c++ )
 {
  // 提取 the c-th color plane
  cvSetImageCOI( timg, c+1 );//设置感兴趣通道
  cvCopy( timg, tgray, 0 );

  // 尝试各种阈值提取得到的（N=11）
  for( l = 0; l < N; l++ )
  {
       // apply Canny. Take the upper threshold from slider
       // Canny helps to catch squares with gradient shading
       if( l == 0 )
       {
        cvCanny( tgray, gray, 0, thresh, 5 );
        //使用任意结构元素膨胀图像
        cvDilate( gray, gray, 0, 1 );
       }
       else
       {
        // apply threshold if l!=0:
        cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );//二值化
       }

       // 找到所有轮廓并且存储在序列中
       cvFindContours( gray, storage, &contours, sizeof(CvContour),
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );

       // 遍历找到的每个轮廓contours
       while( contours )
       {
         //用指定精度逼近多边形曲线
        result = cvApproxPoly( contours, sizeof(CvContour), storage,
         CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );

        if( result->total == 4 &&
         fabs(cvContourArea(result,CV_WHOLE_SEQ)) > 500 &&
         fabs(cvContourArea(result,CV_WHOLE_SEQ)) < 100000 &&
         cvCheckContourConvexity(result) )//是否是凸形
        {
         s = 0;

         for( i = 0; i < 5; i++ )
         {
          // find minimum angle between joint edges (maximum of cosine)
          if( i >= 2 )
          {
           t = fabs(angle(
            (CvPoint*)cvGetSeqElem( result, i ),
            (CvPoint*)cvGetSeqElem( result, i-2 ),
            (CvPoint*)cvGetSeqElem( result, i-1 )));
           s = s > t ? s : t;
          }
         }

         // if 余弦值 足够小，可以认定角度为90度直角
         //cos0.1=83度，能较好的趋近直角
         if( s < 0.1 )
          for( i = 0; i < 4; i++ )
           cvSeqPush( squares,
           (CvPoint*)cvGetSeqElem( result, i ));
        }

        // 继续查找下一个轮廓
        contours = contours->h_next;
       }
  }
 }
 cvReleaseImage( &gray );
 cvReleaseImage( &pyr );
 cvReleaseImage( &tgray );
 cvReleaseImage( &timg );

 return squares;
}

//drawSquares函数用来画出在图像中找到的所有正方形轮廓
void drawSquares( IplImage* img, CvSeq* squares )
{
     CvSeqReader reader;
     IplImage* cpy = cvCloneImage( img );
     int i;
     cvStartReadSeq( squares, &reader, 0 );

     // read 4 sequence elements at a time (all vertices of a square)
//     for( i = 0; i < squares->total; i += 4 )
     for( i = 0; i<2; i += 4 )
     {
          CvPoint pt[4], *rect = pt;
          int count = 4;
          // read 4 vertices
          CV_READ_SEQ_ELEM( pt[0], reader );
          CV_READ_SEQ_ELEM( pt[1], reader );
          CV_READ_SEQ_ELEM( pt[2], reader );
          CV_READ_SEQ_ELEM( pt[3], reader );

          // draw the square as a closed polyline
          cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 2, CV_AA, 0 );
     }

     cvShowImage( wndname, cpy );
     cvReleaseImage( &cpy );
}

//////////////////////////////图像矫正函数-test/////////////////////
void hough(Mat imageSource,CvSeq* squares){
    CvSeqReader reader;
//    IplImage* cpy = cvCloneImage( img );
    int i;
    cvStartReadSeq( squares, &reader, 0 );

    Point2f pt[4];
    // read 4 sequence elements at a time (all vertices of a square)
//    for( i = 0; i < squares->total; i += 4 )
    for( i = 0; i < 2; i += 4 )
    {
         Point2f pt[4], *rect = pt;
         int count = 4;
         // read 4 vertices
         CV_READ_SEQ_ELEM( pt[0], reader );
         CV_READ_SEQ_ELEM( pt[1], reader );
         CV_READ_SEQ_ELEM( pt[2], reader );
         CV_READ_SEQ_ELEM( pt[3], reader );
         // draw the square as a closed polyline
    //     cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 2, CV_AA, 0 );
//         CvPoint P1[4];
    }

    Point2f P2[4];
    Point2f* P1 = pt;
    vector<Point2f>corners;
//         goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
//         for(int i=0;i<corners.size();i++)
//         {
//             circle(DrawLine,corners[i],3,Scalar(255),3);
//             P1[i]=corners[i];
//         }
//         imshow("交点",DrawLine);
    int width=P1[1].x-P1[0].x;
    int hight=P1[2].y-P1[0].y;
    P2[0]=P1[0];
    P2[1]=Point2f(P2[0].x+width,P2[0].y);
    P2[2]=Point2f(P2[0].x,P2[1].y+hight);
    P2[3]=Point2f(P2[1].x,P2[2].y);
    Mat elementTransf;
    elementTransf=  getAffineTransform(P1,P2);
    warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
    imshow("hough",imageSource);
//    cvShowImage( wndname, cpy );
//    cvReleaseImage( &cpy );


//    Point2f P1[4];
//    Point2f P2[4];

}

int main()
{
      storage = cvCreateMemStorage(0);

      img0 = cvLoadImage( "10.jpg", 1 );
      if( !img0 )
      {
       cout <<"不能载入"<<endl;
      }
      img = cvCloneImage( img0 );
      IplImage* img1 = cvCloneImage(img0);
      cvNamedWindow( wndname, 1 );

      // find and draw the squares
      drawSquares( img, findSquares4( img, storage ) );

//      hough(img1,findSquares4( img, storage ));
      cvWaitKey(0);

      cvReleaseImage( &img );
      cvReleaseImage( &img0 );

      cvClearMemStorage( storage );

     cvDestroyWindow( wndname );
     return 0;
}
























