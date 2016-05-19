#include "common_tools.h"
#include "caffe/util/math_functions.hpp"

vector<string> SplitCSVIntoTokens(string line)
{
    vector<string>   result;
    stringstream     lineStream(line);
    string           cell;

    while(std::getline(lineStream, cell, ','))
    {
        result.push_back(cell);
    }
    return result;
}

int read_in_boxesnums(string windows_path,
  vector < vector<box_num> > &boxesnums, vector<string> &img_files)
{
  // clear
  boxesnums.clear();
  img_files.clear();
  // read txt
  ifstream inf(windows_path.c_str());
  char line[256] = "";
  int frame_id = 0;
  int total_number = 0;
  while(inf.getline(line, 256))
  {
    string filename(line);
    img_files.push_back(filename);
    // read boxes number
    int bnum = 0;
    inf.getline(line, 256);
    bnum = atoi(line);
    total_number += bnum;
    //cout << "bnum " <<bnum << endl;
    vector<box_num> bns;
    // read boxes and nums
    for (int i = 0; i < bnum; i++)
    {
      inf.getline(line, 256);
      //cout << line << endl;
      box_num bn;
      sscanf(line, "%f %d %d %d %d", &bn.num, &bn.b.x1, &bn.b.y1, &bn.b.x2, &bn.b.y2);
      bn.b.x1 --;
      bn.b.x2 --;
      bn.b.y1 --;
      bn.b.y2 --;
      bns.push_back(bn);
      // cout << bn.num << " " <<bn.b.x1 << " " <<bn.b.y1 << " " <<bn.b.x2 << " " <<bn.b.y2 <<endl;
      // getchar();
    }
    boxesnums.push_back(bns);
    frame_id ++;
  }
  return total_number;
}

int read_in_boxage_ifa(string annotation,
  vector<box_num> &boxage, vector<string> &img_files)
{
  // clear
  boxage.clear();
  img_files.clear();
  // read txt
  ifstream inf(annotation.c_str());
  char line[256] = "";
  char filename_tmp[256] = "";
  char sex[32] = "";
  char spc[32] = "";
  int total_number = 0;
  while(inf.getline(line, 256))
  {
    total_number += 1;
    box_num bn;
    sscanf(line, "%s %f %s %s %d %d %d %d", filename_tmp, &bn.num, sex, spc,
     &bn.b.x1, &bn.b.y1, &bn.b.x2, &bn.b.y2);
    string filename(filename_tmp);
    img_files.push_back(filename);
    boxage.push_back(bn);
  }
  return total_number;
}

int read_in_boxage_ifa_morph(string annotation,
  vector<box_num> &boxage, vector<string> &img_files)
{
  // clear
  boxage.clear();
  img_files.clear();
  // read txt
  ifstream inf(annotation.c_str());
  char line[256] = "";
  char filename_tmp[256] = "";
  char sex[32] = "";
  char spc[32] = "";
  int total_number = 0;
  int age;
  while(inf.getline(line, 256))
  {
    total_number += 1;
    box_num bn;
    // id_num,picture_num,dob,doa,race,gender,facial_hair,age,age_diff,glasses,photo
    int inttmp;
    char strtmp[256] = "";
    vector <string> vs  = SplitCSVIntoTokens(line);
    // for (int i = 0; i < vs.size(); i ++)
    // {
    //   cout << i << " " << vs[i] << " ";
    // }
    (stringstream(vs[7])) >> age;
    string filename(vs[10]);
    img_files.push_back(filename);

    // read in image
    IplImage *img   = cvLoadImage(filename.c_str());
    bn.b.x1 = 1; bn.b.y1 = 1;
    bn.b.x2 = img->width; bn.b.y2 = img->height;
    bn.num = age;
    // cout << bn.num << " " << bn.b.x1 << " " << bn.b.y1 << " " << bn.b.x2 << " " << bn.b.y2 << " " << filename << endl;
    // getchar();
    cvReleaseImage(&img);
    boxage.push_back(bn);
  }
  return total_number;
}

void crop_sample_in_image_ifa(IplImage *img, box b, int normed_width, int normed_height, unsigned char *str_buffer_img)
{
    CvSize sz;
    sz.height = normed_height;
    sz.width =  normed_width;
    // cout << "check point 4\n";
    // cout << img->nChannels << endl;
    IplImage * crop_img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
    IplImage * debug_crop_img_gray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
    int height = img->height;
    int width = img->width;
    int y1 = std::max(cvRound(b.y1), 0);
    int x1 = std::max(cvRound(b.x1), 0);
    int y2 = std::min(cvRound(b.y2), img->height - 1);
    int x2 = std::min(cvRound(b.x2), img->width - 1);
    CvRect rect;
    rect.x = x1;
    rect.y = y1;
    rect.height = y2 - y1 + 1;
    rect.width = x2 - x1 + 1;
    // cout << "Image " << img->width << " " << img->height <<endl;
    // cout << "Rect "<< rect.x << " "<< rect.y <<" "<< rect.width <<" "<< rect.height << endl;
    // cout << "Boxes "<< b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
    //static int wrong_box_nums = 0;
    // if(b.y2 >= height || b.x2 >= width || b.x1 < 0 || b.y1 < 0)
    // {
    //     cout << "Error Boxes " << wrong_box_nums << " " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
    //     wrong_box_nums ++;
    // }
    cvResetImageROI(img);
    cvSetImageROI(img, rect);
    cvResize(img, crop_img);
    // cvSaveImage("crop_heatmap.bmp", crop_heatmap);
    // gray_crop_img = cvCreateImage(cvGetSize(crop_img), IPL_DEPTH_8U, 1);
    // cvCvtColor(crop_img, gray_crop_img, CV_RGB2GRAY);
    // cout << "Rect " << rect.x << " " << rect.y << " "<< rect.width << " "<< rect.height << endl;
    // cout << "Box  " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
    // cvSaveImage("foo.bmp", crop_img);
    // getchar();
    int width_step = crop_img->widthStep;
    for(int h = 0; h < crop_img->height; h++)
    {
        uchar *debug_p = (uchar *)(debug_crop_img_gray->imageData + h * debug_crop_img_gray->widthStep);
        for(int w = 0; w < crop_img->width; w++)
        {
            unsigned char cB = (unsigned char)crop_img->imageData[h * width_step + w * 3];
            unsigned char cG = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 1];
            unsigned char cR = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 2];
            str_buffer_img[h * normed_width + w] = cR;
            str_buffer_img[1 * normed_height * normed_width + h * normed_width + w] = cG;
            str_buffer_img[2 * normed_height * normed_width + h * normed_width + w] = cB;
            debug_p[w] = (cB + cG + cR) / 3;
        }
    }
    cvReleaseImage(&crop_img);
    // cvSaveImage("debug_foo.png", debug_crop_img_gray);
    // cout << "image saved\n";
    // getchar();
    cvReleaseImage(&debug_crop_img_gray);
}
void data_augment_ifa(IplImage *img, box b, int normed_width, int normed_height, vector<unsigned char*> &str_buffers)
{
  cv::Mat matimg, matblur;
  matimg = cv::Mat(img);
  // origin
  // cout << "Image " << img->width << " " << img->height <<endl;
  b.x1 = b.x1 >= 0? b.x1:0;
  b.y1 = b.y1 >= 0? b.y1:0;
  //cout << "origin\n";
  b.x2 = b.x2 < img->width? b.x2 : img->width;
  b.y2 = b.y2 < img->height? b.y2 : img->height;
  //cout << "check point\n";
  crop_sample_in_image_ifa(img, b, normed_width, normed_height, str_buffers[0]);
  // large head area
  // box bl;
  // bl.x1 = b.x1 - 0.5 * (b.x2-b.x1) >= 0 ? b.x1 - 0.5 * (b.x2-b.x1):0;
  // bl.y1 = b.y1 - 0.5 * (b.y2-b.y1) >= 0 ? b.y1 - 0.5 * (b.y2-b.y1):0;
  // bl.x2 = b.x2 + 0.5 * (b.x2-b.x1) < img->width ? b.x2 + 0.5 * (b.x2-b.x1) : img->width;
  // bl.y2 = b.y2 + 0.5 * (b.y2-b.y1) < img->height ? b.y2 + 0.5 * (b.y2-b.y1) : img->height;
  // crop_sample_in_image_ifa(img, bl, normed_width, normed_height, str_buffers[1]);
  // blur
  //cout << "check point 1\n";
  matblur=matimg.clone();
  GaussianBlur(matimg, matblur, cv::Size(5,5), 0, 0);
  // cv::imshow("blur", matblur);
  // cv::waitKey(20);
  IplImage* imgblur = new IplImage(matblur);;
  //cout << "check point 2\n";
  crop_sample_in_image_ifa(imgblur, b, normed_width, normed_height, str_buffers[2]);
  //crop_sample_in_image_ifa(imgblur, bl, normed_width, normed_height, str_buffers[3]);
  delete imgblur;
  //cout << "check point 3\n";
}

void data_augment_ifa_withoutscale(IplImage *img, box b, int normed_width, int normed_height, vector<unsigned char*> &str_buffers)
{
  cv::Mat matimg, matblur;
  matimg = cv::Mat(img);
  // origin
  // cout << "Image " << img->width << " " << img->height <<endl;
  b.x1 = b.x1 >= 0? b.x1:0;
  b.y1 = b.y1 >= 0? b.y1:0;
  //cout << "origin\n";
  b.x2 = b.x2 < img->width? b.x2 : img->width;
  b.y2 = b.y2 < img->height? b.y2 : img->height;
  //cout << "check point\n";
  //crop_sample_in_image_ifa(img, b, normed_width, normed_height, str_buffers[0]);
  // large head area
  box bl;
  bl.x1 = b.x1 - 0.5 * (b.x2-b.x1) >= 0 ? b.x1 - 0.5 * (b.x2-b.x1):0;
  bl.y1 = b.y1 - 0.5 * (b.y2-b.y1) >= 0 ? b.y1 - 0.5 * (b.y2-b.y1):0;
  bl.x2 = b.x2 + 0.5 * (b.x2-b.x1) < img->width ? b.x2 + 0.5 * (b.x2-b.x1) : img->width;
  bl.y2 = b.y2 + 0.5 * (b.y2-b.y1) < img->height ? b.y2 + 0.5 * (b.y2-b.y1) : img->height;
  crop_sample_in_image_ifa(img, bl, normed_width, normed_height, str_buffers[0]);
  // blur
  //cout << "check point 1\n";
  matblur=matimg.clone();
  GaussianBlur(matimg, matblur, cv::Size(5,5), 0, 0);
  // cv::imshow("blur", matblur);
  // cv::waitKey(20);
  IplImage* imgblur = new IplImage(matblur);;
  //cout << "check point 2\n";
  // crop_sample_in_image_ifa(imgblur, b, normed_width, normed_height, str_buffers[2]);
  crop_sample_in_image_ifa(imgblur, bl, normed_width, normed_height, str_buffers[1]);
  delete imgblur;
  //cout << "check point 3\n";
}

void convert_img_data_ifa(string annotation, 
  int c, int resize_width, int resize_height, 
  string db_path, string backend, 
  float begin_ratio, float end_ratio, bool check_size)
{
    // new db
  scoped_ptr<caffe::db::DB> db1(db::GetDB(backend));
  db1->Open(db_path, caffe::db::NEW);
  scoped_ptr<db::Transaction> txn(db1->NewTransaction());
  Datum datum;

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  int iter_samples = 0;

    // read in images and store them into db
  vector<box_num> boxage;
  vector<string> img_files;
  string value;
  string value2;
  vector<unsigned char *> str_buffer_imgs;
  const int str_buffers_img_size = 2;
  str_buffer_imgs.resize(str_buffers_img_size);
  for (int i = 0; i < str_buffer_imgs.size(); i++)
  {
    str_buffer_imgs[i] = new unsigned char[resize_width * resize_height * 3];
  }
  const int num_samples = read_in_boxage_ifa_morph(annotation, boxage, img_files);
  LOG(INFO) << "total " << num_samples << " samples";
  CHECK_EQ(boxage.size(), img_files.size());
  int begin_index = img_files.size() * begin_ratio;
  int end_index   = img_files.size() * end_ratio;
  LOG(INFO) << db_path << " Starting from " << begin_index \
  << " ending with " << end_index;
  std::vector<int> ages;
  double mu = 0;
  double sigma = 0;
  for (int i = begin_index; i < end_index; i++)
  {
    // read in image
    string img_path = img_files[i];
    IplImage *img   = cvLoadImage(img_path.c_str());
    if (img == 0)
    {
      cout << img_path << endl;
      cout << "error\n";
    }
    string file_stem = get_file_stem(img_files[i]);
    int label = cvRound(boxage[i].num);
    ages.push_back(label);
    mu += label;
    data_augment_ifa_withoutscale (img, boxage[i].b, resize_width, resize_height, str_buffer_imgs);
    for (int j = 0; j < str_buffer_imgs.size(); j++)
    {
      string filestem = get_file_stem(img_files[i]);
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", iter_samples, filestem.c_str());
      // save to datum
      datum.set_channels(c);
      datum.set_height(resize_height);
      datum.set_width(resize_width);
      datum.set_data(str_buffer_imgs[j], resize_width * resize_height * 3);
      datum.set_label(label);

      CHECK(datum.SerializeToString(&value));
      // check
      if (check_size) {
        data_size = datum.channels() * datum.height() * datum.width();
        const std::string& raw_data = datum.data();
        CHECK_EQ(raw_data.size(), data_size) << "Incorrect data field size "
        << raw_data.size();
      }
      // save to db
      txn->Put(string(key_cstr, length), value);
      ++iter_samples;
      if (iter_samples % 5000 == 0 || i == end_index-1) {
        // Commit db
        txn->Commit();
        txn.reset(db1->NewTransaction());
        LOG(INFO) << "[IMAGE]" <<"Processed " << iter_samples << " files " << "total " << num_samples;
        LOG(INFO) << "\tCurrent " << img_files[i] << " Current key " << key_cstr << " label " << boxage[i].num;
      }
      cvReleaseImage(&img);
    }
  }
  for (int i = 0; i < str_buffer_imgs.size(); i++)
  {
    delete []str_buffer_imgs[i];
  }
  // get mu and sigma
  mu /= ages.size();
  for (int i = 0; i < ages.size(); i++)
  {
    sigma += (ages[i] - mu) * (ages[i] - mu);
  }
  sigma /= ages.size();
  LOG(INFO) << "Mu " << mu << "Sigma " << sigma; 
}

void crop_gray_sample_in_image(IplImage *img, box b, int normed_width, int normed_height, unsigned char *str_buffer_img)
{
    CvSize sz;
    sz.height = normed_height;
    sz.width =  normed_width;
    // cout << img_heatmap->nChannels << endl;
    IplImage * crop_img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
    IplImage * debug_crop_img_gray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
    int height = img->height;
    int width = img->width;
    int y1 = std::max(cvRound(b.y1), 0);
    int x1 = std::max(cvRound(b.x1), 0);
    int y2 = std::min(cvRound(b.y2), height - 1);
    int x2 = std::min(cvRound(b.x2), width - 1);
    CvRect rect;
    rect.x = x1;
    rect.y = y1;
    rect.height = y2 - y1 + 1;
    rect.width = x2 - x1 + 1;
    static int wrong_box_nums = 0;
    if(b.y2 >= height || b.x2 >= width || b.x1 < 0 || b.y1 < 0)
    {
        LOG(INFO) << "Error Boxes " << wrong_box_nums << " " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2;
        wrong_box_nums ++;
    }
    cvSetImageROI(img, rect);
    cvResize(img, crop_img);
    // cvSaveImage("crop_heatmap.bmp", crop_heatmap);
    // gray_crop_img = cvCreateImage(cvGetSize(crop_img), IPL_DEPTH_8U, 1);
    // cvCvtColor(crop_img, gray_crop_img, CV_RGB2GRAY);
    // cout << "Rect " << rect.x << " " << rect.y << " "<< rect.width << " "<< rect.height << endl;
    // cout << "Box  " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
    // cvSaveImage("foo.bmp", crop_img);
    // getchar();
    int width_step = crop_img->widthStep;
    for(int h = 0; h < crop_img->height; h++)
    {
        uchar *debug_p = (uchar *)(debug_crop_img_gray->imageData + h * debug_crop_img_gray->widthStep);
        for(int w = 0; w < crop_img->width; w++)
        {
            unsigned char cB = (unsigned char)crop_img->imageData[h * width_step + w * 3];
            unsigned char cG = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 1];
            unsigned char cR = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 2];
            unsigned char gray = (unsigned char) (float(cB) + float(cG) + float(cR)) / 3;
            str_buffer_img[h * normed_width + w] = gray;
            str_buffer_img[1 * normed_height * normed_width + h * normed_width + w] = gray;
            str_buffer_img[2 * normed_height * normed_width + h * normed_width + w] = gray;
            debug_p[w] = (cB + cG + cR) / 3;
        }
    }
    cvReleaseImage(&crop_img);
    // cvSaveImage("debug_foo.png", debug_crop_img_gray);
    // cout << "image saved\n";
    // getchar();
    cvReleaseImage(&debug_crop_img_gray);
}

void crop_sample_in_image(IplImage *img, box b, int normed_width, int normed_height, unsigned char *str_buffer_img)
{
    CvSize sz;
    sz.height = normed_height;
    sz.width =  normed_width;
    // cout << img_heatmap->nChannels << endl;
    IplImage * crop_img = cvCreateImage(sz, IPL_DEPTH_8U, 3);
    IplImage * debug_crop_img_gray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
    int height = img->height;
    int width = img->width;
    int y1 = std::max(cvRound(b.y1), 0);
    int x1 = std::max(cvRound(b.x1), 0);
    int y2 = std::min(cvRound(b.y2), height - 1);
    int x2 = std::min(cvRound(b.x2), width - 1);
    CvRect rect;
    rect.x = x1;
    rect.y = y1;
    rect.height = y2 - y1 + 1;
    rect.width = x2 - x1 + 1;
    static int wrong_box_nums = 0;
    if(b.y2 > height || b.x2 > width || b.x1 < 0 || b.y1 < 0)
    {
        LOG(INFO) << "Error Boxes " << wrong_box_nums << " " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
        wrong_box_nums ++;
    }
    cvSetImageROI(img, rect);
    cvResize(img, crop_img);
    // cvSaveImage("crop_heatmap.bmp", crop_heatmap);
    // gray_crop_img = cvCreateImage(cvGetSize(crop_img), IPL_DEPTH_8U, 1);
    // cvCvtColor(crop_img, gray_crop_img, CV_RGB2GRAY);
    // cout << "Rect " << rect.x << " " << rect.y << " "<< rect.width << " "<< rect.height << endl;
    // cout << "Box  " << b.x1 << " " << b.y1 << " "<< b.x2<< " "<< b.y2<< endl;
    // cvSaveImage("foo.bmp", crop_img);
    // getchar();
    int width_step = crop_img->widthStep;
    for(int h = 0; h < crop_img->height; h++)
    {
        uchar *debug_p = (uchar *)(debug_crop_img_gray->imageData + h * debug_crop_img_gray->widthStep);
        for(int w = 0; w < crop_img->width; w++)
        {
            unsigned char cB = (unsigned char)crop_img->imageData[h * width_step + w * 3];
            unsigned char cG = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 1];
            unsigned char cR = (unsigned char)crop_img->imageData[h * width_step + w * 3 + 2];
            str_buffer_img[h * normed_width + w] = cR;
            str_buffer_img[1 * normed_height * normed_width + h * normed_width + w] = cG;
            str_buffer_img[2 * normed_height * normed_width + h * normed_width + w] = cB;
            debug_p[w] = (cB + cG + cR) / 3;
        }
    }
    cvReleaseImage(&crop_img);
    // cvSaveImage("debug_foo.png", debug_crop_img_gray);
    // cout << "image saved\n";
    // getchar();
    cvReleaseImage(&debug_crop_img_gray);
}

void save_densitymap(vector< vector<float> > &m, string dmap_file)
{
    ofstream outf(dmap_file.c_str(), std::ofstream::out);

    for (int i = 0; i < m.size(); i ++)
    {
        stringstream ss;
        for (int j = 0; j < m[i].size(); j++)
        {
            outf << m[i][j] << " ";
        }
        outf << endl;
    }
    outf.close();
}

void crop_in_densitymap(vector<vector<float> > &m, box b, int normed_width, int normed_height, float *data)
{
    // save_densitymap(m, "m_infun.txt");
    // we need to / 4, because the dmap is 180*320
    int l = b.x1 / 4;
    int t = b.y1 / 4;
    int r = b.x2 / 4;
    int o = b.y2 / 4;
    // cout << "box ";
    // cout << l << " " << t << " " << r << " " << o << endl;
    float rw = (r - l + 0.0) / normed_width;
    float rh = (o - t + 0.0) / normed_height;
    float scale = rw * rh;
    // cout << "scale "<<rw << " " << rh << " " << scale <<endl;
    // getchar();
    static double maxval = 0, minval = 0xFFFF, meanval = 0;
    static double cnt = 0;
    float avg = 0;
    for (int h = 0; h < normed_height; h++)
    {
        for (int w = 0; w < normed_width; w++)
        {
            int ind_dst = h * normed_width + w;
            float ind_src_w = w * rw + l;
            float ind_src_h = h * rh + t;
            int x1 = cvFloor(ind_src_w);
            int x2 = cvCeil(ind_src_w);
            int y1 = cvFloor(ind_src_h);
            int y2 = cvCeil(ind_src_h);
            data[ind_dst] = (y2-ind_src_h) * ((x2-ind_src_w)*m[y1][x1] + (1.0-x2+ind_src_w)*m[y1][x2]) +\
                            (1-y2+ind_src_h) * ((x2-ind_src_w)*m[y2][x1] + (1.0-x2+ind_src_w)*m[y2][x2]);
            data[ind_dst] *= scale;
            if (maxval < data[ind_dst])
              maxval = data[ind_dst];
            if (minval > data[ind_dst])
              minval = data[ind_dst];
            avg += data[ind_dst];
            // cout << ind_src_w << " " << ind_src_h << endl;
            // cout << x1 <<" "<< x2 <<" "<< y1 <<" "<< y2 <<endl;
            // if(m[y1][x1] >= 0.0001)
            // {
            //     cout << "values: " << m[y1][x1] <<" "<< m[y1][x2] <<" "<< m[y2][x1] <<" "<< m[y2][x2] <<endl;
            //     cout << data[ind_dst] <<endl;
            // }
            // getchar();
        }
    }
    avg /= normed_height * normed_width;
    meanval = cnt / (cnt + 1.0) * meanval + 1.0 / (cnt + 1.0) * avg;
    cnt += 1;
    if (int(cnt) % 100000 == 0)
      cout << cnt << " max: " << maxval << " min: " << minval << " avg: " << meanval << endl;;
}

void read_in_densitymap(string dmap_file, vector< vector<float> > &m)
{
    ifstream inf(dmap_file.c_str());
    const int rows = 180;
    const int cols = 320;
    m.clear();
    m.resize(rows);
    for (int i = 0; i < rows; i ++)
        m[i].resize(cols);

    char line[0xFFFF];
    int iter_row = 0;
    while(inf.getline(line, 0xFFFF))
    {
        stringstream ss;
        ss << line;
        for (int i = 0; i < cols; i ++)
        {
            ss >> m[iter_row][i];
        }
        iter_row ++;
    }
    inf.close();
}

string get_file_stem(string &file_path)
{
    int begin = 0;
    int end = file_path.size();
    for(int i = 0; i < file_path.size(); i++)
    {
        if(file_path[i] == '\\' || file_path[i] == '/')
            begin = i + 1;
        if(file_path[i] == '.') // suppose it's the simplest case
            end = i;
    }
    string stem("");
    if (begin < end)
    {
        for(int i = begin; i < end; i++)
        {
            stem = stem + file_path[i];
        }
    }
    return stem;
}
void split_path(string &file_path, string &path, string &name, string &ext)
{
    int begin = 0;
    int end = file_path.size();
    for(int i = 0; i < file_path.size(); i++)
    {
        if(file_path[i] == '\\' || file_path[i] == '/')
            begin = i + 1;
        if(file_path[i] == '.') // suppose it's the simplest case
            end = i;
    }
    name.clear();
    ext.clear();
    path.clear();
    if (begin < end && end < file_path.size()){
        for(int i = begin; i < end; i++)
        {
            name = name + file_path[i];
        }
        for(int i = end; i < file_path.size(); i++)
        {
            ext = ext + file_path[i];
        }
        for(int i = 0; i < begin; i++)
        {
            path = path + file_path[i];
        }
    }
    else
      path = file_path;
}
void save_dmap_to_datum(float * dmap, int w, int h, Datum &datum)
{
    datum.set_channels(1);
    datum.set_height(h);
    datum.set_width(w);
    datum.clear_float_data();
    for (int i = 0; i < w * h; i ++)
        datum.add_float_data(dmap[i]);
}

void save_flatten_dmap_to_datum(float * dmap, int w, int h, Datum &datum)
{
    datum.set_channels(w*h);
    datum.set_height(1);
    datum.set_width(1);
    datum.clear_float_data();
    for (int i = 0; i < w * h; i ++)
        datum.add_float_data(dmap[i]);
}

void convert_img_data(string windows_file_path, string image_fold_path, int c, int resize_width, int resize_height, string db_path, string backend /*= "lmdb"*/, float begin_ratio /*= 0*/, float end_ratio /*= 0.5*/, bool check_size /*= false*/)
{
    // new db
  scoped_ptr<caffe::db::DB> db1(db::GetDB(backend));
  db1->Open(db_path, caffe::db::NEW);
  scoped_ptr<db::Transaction> txn(db1->NewTransaction());
  Datum datum;

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  int iter_samples = 0;

    // read in images and store them into db
  vector < vector<box_num> > boxesnums;
  vector<string> img_files;
  vector<vector<float> > m;
  string value;
  string value2;
  unsigned char * str_buffer_img = new unsigned char[resize_width * resize_height * 3];
  // cout << "11111" << endl;
  const int num_samples = read_in_boxesnums(windows_file_path, boxesnums, img_files);
  LOG(INFO) << "total " << num_samples << "samples";
  int begin_index = img_files.size() * begin_ratio;
  int end_index   = img_files.size() * end_ratio;
  LOG(INFO) << db_path << " Starting from " << begin_index \
  << " ending with " << end_index;
  for (int i = begin_index; i < end_index; i++)
  {
    // read in image
    string img_path = image_fold_path + "/" + img_files[i];
    IplImage *img   = cvLoadImage(img_path.c_str());
    // read in densitymap
    // cout << "33333" << endl;
    string file_stem = get_file_stem(img_files[i]);
    // cout << "44444" << endl;
    for (int j = 0; j < boxesnums[i].size(); j ++)
    {
      int label = cvRound(boxesnums[i][j].num);
      crop_sample_in_image(img, boxesnums[i][j].b, resize_width, resize_height, str_buffer_img);
      // cout << "55555" << endl;
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", iter_samples, img_files[i].c_str());
      // cout << "77777 " << key_cstr << endl;
      // save to datum
      datum.set_channels(c);
      datum.set_height(resize_height);
      datum.set_width(resize_width);
      datum.set_data(str_buffer_img, resize_width * resize_height * 3);
      datum.set_label(label);
      // cout << "88888 " << endl;

      CHECK(datum.SerializeToString(&value));
      // check
      if (check_size) {
          data_size = datum.channels() * datum.height() * datum.width();
          const std::string& raw_data = datum.data();
          CHECK_EQ(raw_data.size(), data_size) << "Incorrect data field size "
              << raw_data.size();
      }
      // save to db
      txn->Put(string(key_cstr, length), value);
      ++iter_samples;
      if (iter_samples % 10000 == 0 || (i == end_index-1 && j == boxesnums[i].size()-1)) {
        // Commit db
        txn->Commit();
        txn.reset(db1->NewTransaction());
        LOG(INFO) << "[IMAGE]" <<"Processed " << iter_samples << " files " << "total " << num_samples;
        LOG(INFO) << "\tCurrent " << img_files[i] << " Current key " << key_cstr << " label " << boxesnums[i][j].num;
      }
    }
    cvReleaseImage(&img);
  }
  delete [] str_buffer_img;
}

void convert_float_label_data(string windows_file_path, string image_fold_path, string db_path, string backend /*= "lmdb"*/, float begin_ratio /*= 0*/, float end_ratio /*= 0.5*/, bool check_size /*= false*/)
{
    // new db
  scoped_ptr<caffe::db::DB> db1(db::GetDB(backend));
  db1->Open(db_path, caffe::db::NEW);
  scoped_ptr<db::Transaction> txn(db1->NewTransaction());
  Datum datum;

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  int iter_samples = 0;

    // read in images and store them into db
  vector < vector<box_num> > boxesnums;
  vector<string> img_files;
  vector<vector<float> > m;
  string value;
  string value2;
  // cout << "11111" << endl;
  const int num_samples = read_in_boxesnums(windows_file_path, boxesnums, img_files);
  LOG(INFO) << "total " << num_samples << "samples";
  int begin_index = img_files.size() * begin_ratio;
  int end_index   = img_files.size() * end_ratio;
  LOG(INFO) << db_path << " Starting from " << begin_index \
  << " ending with " << end_index;
  for (int i = begin_index; i < end_index; i++)
  {
    // read in image
    string img_path = image_fold_path + "/" + img_files[i];
    // read in densitymap
    // cout << "33333" << endl;
    string file_stem = get_file_stem(img_files[i]);
    // cout << "44444" << endl;
    for (int j = 0; j < boxesnums[i].size(); j ++)
    {
      int label = cvRound(boxesnums[i][j].num);
      // cout << "55555" << endl;
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", iter_samples, img_files[i].c_str());
      // cout << "77777 " << key_cstr << endl;
      // save to datum
      datum.set_channels(1);
      datum.set_height(1);
      datum.set_width(1);
      datum.clear_float_data();
      datum.add_float_data(boxesnums[i][j].num);
      datum.set_label(label);
      // cout << "88888 " << endl;

      CHECK(datum.SerializeToString(&value));
      // check
      if (check_size) {
          data_size = datum.channels() * datum.height() * datum.width();
          CHECK_EQ(datum.float_data_size(), data_size) << "Incorrect data field size "
              << datum.float_data_size();
      }
      // save to db
      txn->Put(string(key_cstr, length), value);
      ++iter_samples;
      if (iter_samples % 10000 == 0 || (i == end_index-1 && j == boxesnums[i].size()-1)) {
        // Commit db
        txn->Commit();
        txn.reset(db1->NewTransaction());
        LOG(INFO) << "[FLOAT]" << "Processed " << iter_samples << " files " << "total " << num_samples;
        LOG(INFO) << "\tCurrent " << img_files[i] << " Current key " << key_cstr << " label " << boxesnums[i][j].num;
      }
    }
  }
}

void convert_dmap_data(string windows_file_path, string dmap_fold_path, int c, int dmap_width, int dmap_height, string db_path, string backend /*= "lmdb"*/, float begin_ratio /*= 0*/, float end_ratio /*= 0.5*/, bool check_size /*= false*/)
{
  scoped_ptr<db::DB> db2(db::GetDB(backend));
  db2->Open(db_path, caffe::db::NEW);
  scoped_ptr<db::Transaction> txn2(db2->NewTransaction());
  Datum datum2;
  float * dmap_data = new float[dmap_width * dmap_height];
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  int iter_samples = 0;
  vector < vector<box_num> > boxesnums;
  vector<string> img_files;
  vector<vector<float> > m;
  string value2;
  const int num_samples = read_in_boxesnums(windows_file_path, boxesnums, img_files);
  LOG(INFO) << "total " << num_samples << "samples";

  int begin_index = img_files.size() * begin_ratio;
  int end_index   = img_files.size() * end_ratio;
  LOG(INFO) << db_path << " Starting from " << begin_index \
  << " ending with " << end_index;
  for (int i = begin_index; i < end_index; i++)
  {
    // read in densitymap
    // cout << "33333" << endl;
    string file_stem = get_file_stem(img_files[i]);
    string dmap_path = dmap_fold_path + "/" + file_stem + ".density";
    read_in_densitymap(dmap_path, m);
    // save_densitymap(m, file_stem + ".txt");
    // getchar();
    // for (int a = 0; a < m.size(); a ++)
    // {
    //     for (int b = 0; b < m[a].size(); b++)
    //     {
    //         cout << m[a][b] << " ";
    //     }
    //     cout << endl;
    //     getchar();
    // }
    // cout << "44444" << endl;
    for (int j = 0; j < boxesnums[i].size(); j ++)
    {
      int label = cvRound(boxesnums[i][j].num);
      // cout << "55555" << endl;
      // cout << "label " << label << endl;
      crop_in_densitymap(m, boxesnums[i][j].b, dmap_width, dmap_height, dmap_data);
      // cout << "66666" << endl;
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", iter_samples, img_files[i].c_str());
      // cout << "77777 " << key_cstr << endl;
      // save to datum
      // cout << "88888 " << endl;
      save_dmap_to_datum(dmap_data, dmap_width, dmap_height, datum2);
      datum2.set_label(label);
      // static bool fetched = 0;
      // // if (label == 5 && fetched == 0)
      // // {
      // //   cout << boxesnums[i][j].b.x1 << " "<<boxesnums[i][j].b.y1 << " "<<boxesnums[i][j].b.x2 << " "<< boxesnums[i][j].b.y2 <<endl;
      // //   cout << key_cstr << endl;
      // //   fetched = 1;
      // //   for (int a = 0; a < dmap_width * dmap_height; a ++)
      // //   {
      // //       if(a % dmap_width == 0)
      // //           cout << endl;
      // //       cout << dmap_data[a] << " ";
      // //   }
      // //   getchar();
      // // }
      CHECK(datum2.SerializeToString(&value2));
      // check
      if (check_size) {
          data_size = datum2.channels() * datum2.height() * datum2.width();
          CHECK_EQ(datum2.float_data_size(), data_size) << "Incorrect data field size "
              << datum2.float_data_size();
      }
      // save to db
      txn2->Put(string(key_cstr, length), value2);
      ++iter_samples;
      if (iter_samples % 10000 == 0 || (i == end_index-1 && j == boxesnums[i].size()-1)) {
        // Commit db
        txn2->Commit();
        txn2.reset(db2->NewTransaction());
        LOG(INFO) << "[D-MAP]" <<"Processed " << iter_samples << " files " << "total " << num_samples;
        LOG(INFO) << "\tCurrent " << img_files[i] << " Current key " << key_cstr << " label " << boxesnums[i][j].num;
      }
    }
  }
  delete [] dmap_data;
}

void convert_img_data_ped(string prefix,
  string annotation, 
  int c, int resize_width, int resize_height, 
  string db_path, string backend, 
  float begin_ratio, float end_ratio)
{
  // new db
  scoped_ptr<caffe::db::DB> db_data(db::GetDB(backend));
  scoped_ptr<caffe::db::DB> db_data_bbrgs(db::GetDB(backend));
  db_data->Open(db_path, caffe::db::NEW);
  db_data_bbrgs->Open(db_path+"_bbrgs", caffe::db::NEW);
  scoped_ptr<db::Transaction> txn(db_data->NewTransaction());
  scoped_ptr<db::Transaction> txn_bbrgs(db_data_bbrgs->NewTransaction());
  Datum datum;
  Datum datum_bbrgs;

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int iter_samples = 0;

    // read in images and store them into db
  vector<vector<box_num> > boxage;
  vector<vector<box_num> > boxage1;
  vector<string> img_files;
  string value;
  string value1;
  unsigned char * str_buffer_img;
  str_buffer_img = new unsigned char[resize_width * resize_height * c];
  const int num_samples = read_in_annotations_ped(annotation, boxage, boxage1, img_files);
  LOG(INFO) << "total " << num_samples << " samples";
  CHECK_EQ(boxage.size(), img_files.size());
  CHECK_EQ(boxage.size(), boxage1.size());
  int begin_index = img_files.size() * begin_ratio;
  int end_index   = img_files.size() * end_ratio;
  LOG(INFO) << db_path << " Starting from " << begin_index \
  << " ending with " << end_index;
  vector<int> data_num(2, 0);
  for (int i = begin_index; i < end_index; i++)
  {
    // read in image
    string img_path = img_files[i];
    if (prefix != ""){
      string path, name, ext;
      split_path(img_path, path, name, ext);
      img_path = prefix + "/" + name + ext;
    }
    IplImage *img   = cvLoadImage(img_path.c_str());
    if (img == 0){
      LOG(INFO) << img_path << " not exist";
    }
    string filestem = get_file_stem(img_path);
    // LOG(INFO) << boxage[i].size();
    for (int j = 0; j < boxage[i].size(); j++)
    {
      box_num &bn = boxage[i][j];
      box_num &bn1 = boxage1[i][j];
      // LOG(INFO) << bn.str();
      // jump over some cases
      if (bn.b.y2 - bn.b.y1 < 50){
        continue;
      }
      if (bn.b.x1 < 0 || bn.b.y1 < 0 || bn.b.x2 >= img->width || bn.b.y1 >= img->height
        || bn1.b.x1 < 0 || bn1.b.y1 < 0 || bn1.b.x2 >= img->width || bn1.b.y1 >= img->height)
      {
        continue;
      }
      // enlarge box
      // crop
      crop_sample_in_image(img, bn.b, resize_width, resize_height, str_buffer_img);
      int label = cvRound(bn.num);
      label = label == 11 ? 1 : 0;

      double r = (0.0 + caffe::caffe_rng_rand()) / UINT_MAX;
      if (r > 0.98 && label != 1 || label == 1){
        data_num[label] ++;
        int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", iter_samples, filestem.c_str());
        // save to datum
        datum.set_channels(c);
        datum.set_height(resize_height);
        datum.set_width(resize_width);
        datum.set_data(str_buffer_img, resize_width * resize_height * c);
        datum.set_label(label);
        // bb rgs
        if(label == 1){ // pedestrian
          bn1.b.x1 = bn.b.x1 - bn1.b.x1;
          bn1.b.y1 = bn.b.y1 - bn1.b.y1;
          bn1.b.x2 = bn.b.x2 - bn1.b.x2;
          bn1.b.y2 = bn.b.y2 - bn1.b.y2;
        }
        datum_bbrgs.set_channels(4);
        datum_bbrgs.set_height(1); 
        datum_bbrgs.set_width(1);
        datum_bbrgs.clear_float_data();
        datum_bbrgs.add_float_data(bn1.b.x1);
        datum_bbrgs.add_float_data(bn1.b.y1);
        datum_bbrgs.add_float_data(bn1.b.x2);
        datum_bbrgs.add_float_data(bn1.b.y2);

        CHECK(datum.SerializeToString(&value));
        CHECK(datum_bbrgs.SerializeToString(&value1));
        // check
        int data_size = datum.channels() * datum.height() * datum.width();
        int data_size1 = datum_bbrgs.channels() * datum_bbrgs.height() * datum_bbrgs.width();
        CHECK_EQ(datum.data().size(), data_size) << "Incorrect data field size "
        << datum.data().size();
        // LOG(INFO) << "Check " << value1;
        // CHECK_EQ(datum_bbrgs.data().size(), data_size1) << "Incorrect data field size "
        // << datum_bbrgs.data().size();
        // save to db
        txn->Put(string(key_cstr, length), value);
        txn_bbrgs->Put(string(key_cstr, length), value1);
        ++iter_samples; 
      }
      
      if (iter_samples % 1000 == 0 || i == end_index-1 && j == boxage[i].size()-1) {
        // Commit db
        txn->Commit();
        txn_bbrgs->Commit();
        txn.reset(db_data->NewTransaction());
        txn_bbrgs.reset(db_data_bbrgs->NewTransaction());
        if (iter_samples % 1000 == 0){
          LOG(INFO) << "[IMAGE]" <<"Processed " << iter_samples << " files " << "total " << num_samples;
          LOG(INFO) << "\tCurrent " << img_path << " Current key " << key_cstr <<" "<< bn.str();
        }
      }
    }
    cvReleaseImage(&img);
  }
  LOG(INFO) << "data nums: " << data_num[0] << " " << data_num[1] ;
    delete [] str_buffer_img;
}
int read_in_annotations_ped(string annotation,
  vector<vector<box_num> > &box_ids, vector<vector<box_num> > &box_gts, 
  vector<string> &img_files)
{
  // clear
  box_ids.clear();
  box_gts.clear();
  img_files.clear();
  // read txt
  ifstream inf(annotation.c_str());
  char line[256] = "";
  int frame_id = 0;
  int total_number = 0;
  double overlap = 0;
  while(inf.getline(line, 256))
  {
    string filename(line);
    img_files.push_back(filename);
    // read boxes number
    // LOG(INFO) << "Check " << filename;
    int bnum = 0;
    inf.getline(line, 256);
    bnum = atoi(line);
    // LOG(INFO) << "Check " << bnum;
    //cout << "bnum " <<bnum << endl;
    vector<box_num> bns;
    vector<box_num> bn1s;
    // read boxes and nums
    for (int i = 0; i < bnum; i++)
    {
      inf.getline(line, 256);
      //cout << line << endl;
      box_num bn;
      box_num bn1;
      // % x, y, w, h, scroe, class, gx, gy, gw, gh, oa
      vector<string> vs = SplitCSVIntoTokens(line);
      stringstream(vs[5]) >> bn.num;
      stringstream(vs[0]) >> bn.b.x1;
      stringstream(vs[1]) >> bn.b.y1;
      stringstream(vs[2]) >> bn.b.x2;
      stringstream(vs[3]) >> bn.b.y2;

      stringstream(vs[5]) >> bn1.num;
      stringstream(vs[6]) >> bn1.b.x1;
      stringstream(vs[7]) >> bn1.b.y1;
      stringstream(vs[8]) >> bn1.b.x2;
      stringstream(vs[9]) >> bn1.b.y2;
      stringstream(vs[9]) >> overlap;
      // caltech resonable set
      if ((bn1.b.y2 < 50 && bn.num == 11) || (bn.num == 10 && overlap < 0.5)){
        continue;
      }
      // if ((bn.b.y2 < 50 && bn.num == 11)){
      //   continue;
      // }
      bn.b.x2 = bn.b.x2 + bn.b.x1 -1;  
      bn.b.y2 = bn.b.y2 + bn.b.y1 -1;
      bn1.b.x2 = bn1.b.x2 + bn1.b.x1 -1;  
      bn1.b.y2 = bn1.b.y2 + bn1.b.y1 -1;
      bns.push_back(bn);
      bn1s.push_back(bn1);
      total_number ++;
      // LOG(INFO) << bn.str() << "\t" << bn1.str();
      // getchar();
    }
    box_ids.push_back(bns);
    box_gts.push_back(bn1s);
    frame_id ++;
  }
  return total_number;
}