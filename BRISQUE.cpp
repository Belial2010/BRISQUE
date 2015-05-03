#include  <fstream>
#include  <vector>
#include  <iostream>
#include  <iomanip>
#include  <opencv2/opencv.hpp>

 using namespace std;
 using namespace cv;

double brisquescore(Mat image);
void brisque_feature(Mat imdist,vector<double> &feat);
void estimateggdparam(Mat vec,double &gamparam,double &sigma);
void estimateaggdparam(Mat vec,double &alpha,double &leftstd,double &rightstd);
void brisque_process(Mat imdist,vector<double> &feat);
Mat circshift(Mat structdis,int a,int b);
double Gamma(double x);

int _tmain(int argc, _TCHAR* argv[])
{
	Mat image= imread("5616_3744.jpg");
	if (image.empty())
	{
		cout<<"img read fail"<<endl;
	}
	//As the image itself beyond the opencv limit, we are divided into four parts:
	//image quality evaluation and then calculate the average
	//Output: A quality score of the image. The score typically has a value
    //              between 0 and 100 (0 represents the best quality, 100 the worst).
	double t1;
	Rect tmp1;
	tmp1.x=0;
	tmp1.y=0;
	tmp1.width=image.cols/2;
	tmp1.height=image.rows/2;
	t1 = (double)cvGetTickCount();
	double score1=brisquescore(image(tmp1));
	cout<<"first part image quality evaluation score="<<score1<<endl;
	cout <<"cost time=" << t1 / ((double)cvGetTickFrequency()*1000000.) <<" s"<< endl;
	
	tmp1.x=image.cols/2;
	t1= (double)cvGetTickCount();
	double score2=brisquescore(image(tmp1));
	cout<<"second part image quality evaluation score="<<score2<<endl;
	cout << "cost time=" << t1 / ((double)cvGetTickFrequency()*1000000.) <<" s"<< endl;

	tmp1.y=image.rows/2;
	t1= (double)cvGetTickCount();
	double score3=brisquescore(image(tmp1));
	cout<<"third part image quality evaluation score="<<score3<<endl;
	cout <<"cost time"=" << t1 / ((double)cvGetTickFrequency()*1000000.) <<" s"<< endl;

	tmp1.x=0;
	t1= (double)cvGetTickCount();
	double score4=brisquescore(image(tmp1));
	cout<<"forth part image quality evaluation score="<<score4<<endl;
	cout << "cost time=" << t1/ ((double)cvGetTickFrequency()*1000000.) <<" s"<< endl;

	double score_mean=(score1+score2+score3+score4)/4;
	cout<<"image quality evaluation score="<<score_mean<<endl;
	cout<<"image quality evaluation end"<<endl;

	cin.get();
	return 0;
}

double  brisquescore(Mat image)
{
	if (image.empty())
	{
		cout<<"img read fail"<<endl;
	}
	Mat gray;
	if (image.channels()==3)
	{
		cvtColor(image,gray,CV_RGB2GRAY);//
	} 
	else
	{
		gray=image.clone();
	}
	gray.convertTo(gray, CV_64FC1);//char2double
	//imwrite("gray1600.png",gray);
	/*imshow("gray",gray);
	waitKey(0);*/
	//cout<<gray.at<double>(0,0)<<endl;
	vector<double> feat;
	brisque_feature(gray,feat);
	//cout<<"feat computed"<<endl;

	/*---------------------------------------------------------------------*/
	//Quality Score Computation
	/*---------------------------------------------------------------------*/
	

	const char* filename="test_ind";
	ofstream out(filename);
	if(out.is_open()) {
		out<<"1 ";
		for (unsigned int i=0;i<feat.size();i++)
		{
			out<<i+1<<":"<<setiosflags(7)<<feat[i]<<" ";
			//setiosflags(ios::fixed)<<setprecision(6)<<
		}
	}

	system("del output test_ind_scaled dump");//del output,test_ind_scaled,dump
	system("svm-scale -r allrange test_ind>> test_ind_scaled");
	system("svm-predict -b 1 test_ind_scaled allmodel output >>dump");
	

	ifstream in("output");
	double score;
	if (in.is_open())
	{
		in>>score;
	}

	return score;
}

void brisque_feature(Mat imdist,vector<double> &feat)
{
	if (imdist.empty())
	{
		cout<<"img read fail"<<endl;
	}
   //------------------------------------------------
   //Feature Computation
   //-------------------------------------------------
	
	int scalenum = 2;
	feat.clear();
	Mat imdisthalf=Mat::zeros(imdist.rows/2,imdist.cols/2,imdist.type());
	if (2==scalenum)
	{
		brisque_process(imdist,feat);
		resize(imdist,imdisthalf,Size(imdist.cols/2,imdist.rows/2));
		//cout<<imdisthalf.at<double>(0,0)<<endl;
		//for (unsigned int i=0;i<feat.size();i++)
		//{
		//	cout<<feat[i]<<endl;
		//}
		brisque_process(imdisthalf,feat);
	}
		
}

void estimateggdparam(Mat vec,double &gamparam,double &sigma)
{
	
   Mat vec2=vec.clone();
   //cout<<vec2.at<double>(0,0)<<endl;
   Scalar sigma_sq=mean(vec2.mul(vec2));
   sigma=sqrt(sigma_sq[0]);
   Scalar E=mean(abs(vec));
   double rho=sigma_sq[0]/(E[0]*E[0]);

   vector<double> gam;
   vector<double> r_gam;
   vector<double> rho_r_gam;
   unsigned int number=int((10-0.2f)/0.001f)+1;
   gam.clear();
   r_gam.clear();
   rho_r_gam.clear();
   gam.resize(number);
   r_gam.resize(number);
   rho_r_gam.resize(number);
   
   for(unsigned i=0;i<number;i++)
   {
	   if (0==i)
	   {
		   gam[i]=0.2;
	   }else
	   {
		    gam[i]=gam[i-1]+0.001f;
	   }
	   r_gam[i]= (Gamma(1.f/gam[i])*Gamma(3.f/gam[i]))/(Gamma(2./gam[i])*Gamma(2./gam[i]));
	   rho_r_gam[i]=abs(rho-r_gam[i]);
   }
   //find min and pos
   //min_element(dv.begin(),dv.end()) return vector<double>::iterator, as location of point

   int pos = (int) ( min_element(rho_r_gam.begin(), rho_r_gam.end()) -  rho_r_gam.begin() );
   //gamma
   gamparam=gam[pos];
}

void estimateaggdparam(Mat vec,double &alpha,double &leftstd,double &rightstd)
{

	vector<double> left;
	vector<double> right;
	left.clear();
	right.clear();
	for (unsigned int i=0;i<vec.rows;i++)
	{
		double *data1=vec.ptr<double>(i);
		for (unsigned int j=0;j<vec.cols;j++)
		{
			if (/*vec.at<double>(i,j)<0*/data1[j]<0)
			{
				left.push_back(/*vec.at<double>(i,j)*/data1[j]);
			}
			else if (/*vec.at<double>(i,j)>0*/data1[j]>0)
			{
				right.push_back(/*vec.at<double>(i,j)*/data1[j]);
			}
		}
	}
	for (unsigned int i=0;i<left.size();i++)
	{
		left[i]=left[i]*left[i];
	}
	for (unsigned int i=0;i<right.size();i++)
	{
		right[i]=right[i]*right[i];
	}
	double leftsum=0.f;
	for (unsigned int i=0;i<left.size();i++)
	{
		leftsum+=left[i];
	}
	double rightsum=0.f;
	for (unsigned int i=0;i<right.size();i++)
	{
		rightsum+=right[i];
	}
	leftstd=sqrt(leftsum/left.size());//mean
	rightstd =sqrt(rightsum/right.size());//mean
	double gammahat           = leftstd/rightstd;
	Mat vec2;
	multiply(vec,vec,vec2);
	Scalar tmp1=mean(abs(vec));
	Scalar tmp2=mean(vec2);
	double rhat=tmp1[0]*tmp1[0]/tmp2[0];
	
	double rhatnorm=(rhat*(gammahat*gammahat*gammahat +1)*(gammahat+1))/((gammahat*gammahat +1)*(gammahat*gammahat +1));

	vector<double> gam;
	vector<double> r_gam;
	vector<double> r_gam_rha;
	unsigned int number=int((10-0.2f)/0.001f)+1;
	gam.resize(number);
	r_gam.resize(number);
	r_gam_rha.resize(number);
	
	for(unsigned i=0;i<number;i++)
	{
		if (0==i)
		{
			gam[0]=0.2;
		} 
		else
		{
			gam[i]=gam[i-1]+0.001f;
		}
		
		r_gam[i]=(Gamma(2.f/gam[i])*Gamma(2.f/gam[i]))/(Gamma(1./gam[i])*Gamma(3./gam[i]));
		r_gam_rha[i]=(r_gam[i]-rhatnorm)*(r_gam[i]-rhatnorm);
	}


	//find min and pos
	int pos = (int) ( min_element(r_gam_rha.begin(),r_gam_rha.end()) - r_gam_rha.begin() );
	alpha = gam[pos];
}

void brisque_process(Mat imdist,vector<double> &feat)
{
	if (imdist.empty())
	{
		cout<<"img read fail"<<endl;
	}
	Mat mu,mu_sq;
	Mat sigma=Mat::zeros(imdist.rows,imdist.cols,imdist.type());
	Mat imgdouble;

	Mat imdist_mu;
	Mat avoidzero;
	double alpha,overallstd;
	Mat structdis;
	
	//Ptr<FilterEngine> f= createGaussianFilter( CV_64FC1, Size(3,3), 1, 1);
	//Mat tmp1=getGaussianKernel(3,1);
	//Ptr<FilterEngine> f= createSeparableLinearFilter(CV_64FC1,CV_64FC1, tmp1, tmp1,Point(-1,-1), 0);
	//Mat tmp2=Mat::ones(3,3,CV_64FC1);
	//f->apply(tmp2,tmp2);
	//cout<<tmp2<<endl;
	GaussianBlur(imdist,mu,Size(7,7),7.f/6,7.f/6,0);
	//BORDER_CONSTANT=0
	//cout<<imdist.at<double>(0,0)<<endl;
	//cout<<mu.at<double>(0,0)<<endl;
	//cout<<mu.at<double>(0,1)<<endl;
	multiply(mu,mu,mu_sq);
	//cout<<setprecision(10)<<mu_sq.at<double>(0,0)<<endl;
	//Mat imdist1=imdist.clone();
	//cout<<imdist1.at<double>(0,0)<<endl;
	//Mat mu1=mu.clone();
	//Mat imdist_mu1;
 //   subtract(imdist1,mu1,imdist_mu1);
	//imdist_mu1.mul(imdist_mu1);
	//GaussianBlur(imdist_mu1,imdist_mu1,Size(7,7),7.f/6,7.f/6,0);
	//cout<<imdist_mu1.at<double>(0,0)<<endl;

	multiply(imdist,imdist,imgdouble);

	GaussianBlur(imgdouble,imgdouble,Size(7,7),7.f/6,7.f/6,0);
	//cout<<setprecision(10)<<imgdouble.at<double>(0,0)<<endl;
	

	for (unsigned int i=0;i<imgdouble.rows;i++)
	{
		double *data1=imgdouble.ptr<double>(i);
		double *data2=mu_sq.ptr<double>(i);
		double *data3=sigma.ptr<double>(i);
		for (unsigned int j=0;j<imgdouble.cols;j++)
		{
			data3[j]=sqrt(abs(data1[j]-data2[j]));
			//sigma.at<double>(i,j)=sqrt(abs(imgdouble.at<double>(i,j)-mu_sq.at<double>(i,j)));
		}
	}

	subtract(imdist,mu,imdist_mu);
	avoidzero=Mat::ones(sigma.rows,sigma.cols,sigma.type());
	add(sigma,avoidzero,sigma);

	//avoidzero=Mat::ones(imdist_mu1.rows,imdist_mu1.cols,imdist_mu1.type());
	//add(imdist_mu1,avoidzero,imdist_mu1);
	//divide(imdist_mu,imdist_mu1,structdis);
	
	divide(imdist_mu,sigma,structdis);//.......................................equation 1
	//imshow("str",structdis);
	//waitKey(1000);

	//cout<<setprecision(10)<<structdis.at<double>(0,0)<<endl;

	estimateggdparam(structdis,alpha,overallstd);

	feat.push_back(alpha);

	feat.push_back(overallstd*overallstd);

	Mat shifted_structdis;
	Mat pair;
	double constvalue,meanparam,leftstd,rightstd;
	int shifts[][2]={0,1,1,0,1,1,-1,1};
	
	for(unsigned int  itr_shift =0;itr_shift <4;itr_shift ++)
	{
		
		shifted_structdis=circshift(structdis,shifts[itr_shift][0],shifts[itr_shift][1]);
		//cout<<setprecision(10)<<shifted_structdis.at<double>(0,0)<<endl;
		pair=structdis.mul(shifted_structdis);//.............................................................................equation 7,8,9,10
		//cout<<pair.at<double>(0,0)<<endl;
		estimateaggdparam(pair,
			                             alpha,
			                             leftstd,//left sigma
										 rightstd//right sigma
										 );
		
		constvalue=(sqrt(Gamma(1/alpha))/sqrt(Gamma(3/alpha)));
		meanparam=(rightstd-leftstd)*(Gamma(2/alpha)/Gamma(1/alpha))*constvalue;//.................equation 15

		feat.push_back(alpha);
		feat.push_back(meanparam);
		feat.push_back(leftstd*leftstd);
		feat.push_back(rightstd*rightstd);
	}
}

Mat circshift(Mat structdis,int a,int b)
{
	/*
	A = [ 1 2 3;
	         4 5 6; 
			 7 8 9];
	B=circshift(A,[0,1])

		B =

		3     1     2
		6     4     5
		9     7     8

		K>> B=circshift(A,[1,0])

		B =

		7     8     9
		1     2     3
		4     5     6

		K>> B=circshift(A,[-1,0])

		B =

		4     5     6
		7     8     9
		1     2     3

	*/
	
	Mat shiftx=Mat::zeros(structdis.rows,structdis.cols,structdis.type());
	if (0==a)
	{//unchanged 
		shiftx=structdis.clone();
	}
	else if(1==a)
	{//		
		for (unsigned int i=0;i<structdis.rows-1;i++)
		{
			for (unsigned int j=0;j<structdis.cols;j++)
			{
				shiftx.at<double>(i+1,j)=structdis.at<double>(i,j);
			}
		}
			for (unsigned int j=0;j<structdis.cols;j++)
		          shiftx.at<double>(0,j)=structdis.at<double>(structdis.rows-1,j);
	}
	else if (-1==a)
	{
		for (unsigned int i=0;i<structdis.rows-1;i++)
		{
			for (unsigned int j=0;j<structdis.cols;j++)
			{
				shiftx.at<double>(i,j)=structdis.at<double>(i+1,j);
			}
		}
		for (unsigned int j=0;j<structdis.cols;j++)
			shiftx.at<double>(structdis.rows-1,j)=structdis.at<double>(0,j);
	}
	/*
	K>>  A = [ 1 2 3;4 5 6; 7 8 9];
	K>>  B=circshift(A,[0,1])

		B =

		3     1     2
		6     4     5
		9     7     8
		*/
	Mat shifty=Mat::zeros(shiftx.rows,shiftx.cols,shiftx.type());
	if (0==b)
	{
		shifty=shiftx.clone();
	}
	else if (1==b)
	{
		for (unsigned int i=0;i<shiftx.rows;i++)
		{
			for (unsigned int j=0;j<shiftx.cols-1;j++)
			{
				shifty.at<double>(i,j+1)=shiftx.at<double>(i,j);
			}
		}
		for (unsigned int i=0;i<shiftx.rows;i++)
			shifty.at<double>(i,0)=shiftx.at<double>(i,shiftx.cols-1);
	}


	return shifty;
}

double Gamma( double x )
{//x>0
	if( x > 2 && x<= 3 )
	{
		const double c0 =  0.0000677106;
		const double c1 = -0.0003442342;
		const double c2 =  0.0015397681;
		const double c3 = -0.0024467480;
		const double c4 =  0.0109736958;
		const double c5 = -0.0002109075;
		const double c6 =  0.0742379071;
		const double c7 =  0.0815782188;
		const double c8 =  0.4118402518;
		const double c9 =  0.4227843370;
		const double c10 = 1.0000000000;
		double temp = 0;
		temp = temp + c0*pow( x-2.0, 10.0) + c1*pow( x-2.0, 9.0);
		temp = temp + c2*pow( x-2.0, 8.0) + c3*pow( x-2.0 , 7.0);
		temp = temp + c4*pow( x-2.0, 6.0) + c5*pow( x-2.0, 5.0 );
		temp = temp + c6*pow( x-2.0, 4.0 ) + c7*pow( x-2.0, 3.0 );
		temp = temp + c8*pow( x-2.0, 2.0 ) + c9*( x-2.0) + c10;
		return temp;
	}
	else if( x>0 && x<=1 )
	{
		return Gamma( x+2 )/(x*(x+1) );
	}
	else if( x > 1 && x<=2 )
	{
		return Gamma( x+1 )/x;
	}
	else if( x > 3 )
	{
		int i = 1;
		double temp = 1;
		while( ((x-i)>2 && (x-i) <= 3 ) == false )
		{
			temp = (x-i) * temp;
			i++;
		}
		temp = temp*(x-i);
		return temp*Gamma( x-i);
	}
	else
	{
		return 0;
	}
}
