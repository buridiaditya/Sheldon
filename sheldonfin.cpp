#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<stack>
#include"opencv2/imgproc/imgproc.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/core/core.hpp"
#include<opencv2/features2d/features2d.hpp>
#include<typeinfo>
#include"opencv/cv.h"
#include"opencv2/ml/ml.hpp"
#include<fstream>
#include<algorithm>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <termios.h>
#include<stdio.h>
#include <tesseract/baseapi.h>

using namespace std;
using namespace cv;

vector<char> numbers;
vector<char> ops;

typedef struct point{
	int x;
	int y;
} point;
//audrino
int fd;
void sendCommand(const char* command) {
	write(fd, command, 1);
	cout << "sending " << command[0] << endl;
}
//classification
void classify(vector<char> oper){
	for(int i=0;i<oper.size();i++){
		switch(oper[i]){			
			case '+':ops.push_back('+');break;
			case '-': case 0 :ops.push_back('-');break;
			case '/': case 'l' :ops.push_back('/');break;
			case '*': case 'x': case 'X' :ops.push_back('*');break;
			default:numbers.push_back(oper[i]);break;
		}
	}
}
//caluculation of postfix
class postfix{
private:
	stack<float> num;
	float nn;
	int s;
	char su;
public:
	postfix();
	void calculate(vector<char> str);
	float show();
};
postfix::postfix(){
	s=0;
}
void postfix::calculate(vector<char> str)
{
	float n1, n2, n3;
	while (s<str.size()){
		su=str[s];
		if ((su)>'0' && (su<='0'+9) )
		{
			nn = su - '0';
			num.push(nn);
		}
		else
		{
			n1 = num.top();
			num.pop();
			n2 = num.top();
			num.pop();
			switch (su)
			{
			case '+':
				n3 = n2 + n1;
				break;
			case '-':
				n3 = n2 - n1;
				break;
			case '/':
				n3 = n2 / n1;
				break;
			case '*':
				n3 = n2 * n1;
				break;
			case '^':
				n3 = pow(n2, n1);
				break;
			default:
				cout << "Unknown operator";
			}

			num.push(n3);
		}
		s++;
	}
}
float postfix::show(){
	nn = num.top();
	num.pop();
	return nn;
}
//preference
int preference(char inpu){
	switch(inpu){
		case '+':case '-':return 1;break;
		case '*':case '/':return 2;break;
	}
}
//infix to postfix
vector<char> infixtopostfix(vector<char> inpu){
	stack<char> store;
	vector<char> postfix(0);
	for(int i=0;i<inpu.size();i++){
		if(inpu[i]<='0'+9 && inpu[i]>'0'){
			postfix.push_back(inpu[i]);
		}
		else{
			if(store.size()==0){
				store.push(inpu[i]);
			}
			else if(preference(store.top())<=preference(inpu[i]) ){
				store.push(inpu[i]);
			}
			else{
				do{
				postfix.push_back(store.top());
				store.pop();
				}while(store.size()!=0);
				store.push(inpu[i]);
			}
		}
	}
	while(store.size()){
		postfix.push_back(store.top());
		store.pop();
	}
	return postfix;
}

int main(){
	fd = open("/dev/ttyACM0", O_RDWR | O_NOCTTY);  //Opening device file
    //printf("fd opened as %i\n", fd);
	int waste,value;
	char det;
	float botangle;
	float charangle;
	Mat feed,feed0;
	VideoCapture cap("1");	
	point cor;
	point bothead;
	point bottail;
	vector<point> loca;
	vector<char> oper;
	vector<char> resexp;
	vector<point> output;
	vector<vector<Point> > contours;
	vector<Rect> rectangle;
	//taking input from camera and filtering and creating output img
	feed0=imread("test.jpg",CV_LOAD_IMAGE_COLOR);
	Mat outp(feed0.rows,feed0.cols,CV_8UC1);
	for(int i=0;i<feed0.rows;i++){
		for(int j=0;j<feed0.cols;j++){
			if((feed0.at<Vec3b>(i,j)[0]>210) && (feed0.at<Vec3b>(i,j)[1]>210) && (feed0.at<Vec3b>(i,j)[2]>210)){
				outp.at<uchar>(i,j)=255;
			}
			else{
				outp.at<uchar>(i,j)=0;					
			}
		}
	}
	imwrite("out0.jpg",outp);
	findContours(outp,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point(0,0));
	for(int i=0;i<contours.size();i++){
		if(contourArea(contours[i])>1000){
		Rect react=boundingRect(contours[i]);
		rectangle.push_back(react);
		}
	}
	Mat outp1=imread("out0.jpg",0);
	Point a,b;
	for(int k=0;k<rectangle.size();k++){
		Mat character(45,45,CV_8UC1,Scalar(255,255,255));
		a=rectangle[k].tl();
		b=rectangle[k].br();
		for(int i=a.y+2;i<=b.y-2;i++){
			for(int j=a.x+2;j<=b.x-2;j++){
				character.at<uchar>(i-rectangle[k].y,j-rectangle[k].x) = outp1.at<uchar>(i,j);
			}
		}
		imwrite("tcat.jpg",character);
		tesseract::TessBaseAPI tess;
		tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
		tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
		tess.SetImage((uchar*)character.data, character.cols,character.rows, 1, character.cols);
		char* out = tess.GetUTF8Text();
		if((out[0]<='0'+9 && out[0]>'0') || out[0]=='+' || out[0]=='*'|| out[0]=='/'|| out[0]=='-' || out[0]=='x' || out[0]=='X'|| int(out[0])==0 ){
		oper.push_back(out[0]);
		cor.x=(a.x+b.x)/2;
		cor.y=(a.y+b.y)/2;
		loca.push_back(cor);
		}
	}
	classify(oper);
	//algorithm for max
	vector<char> numcopy=numbers;
	vector<char> opscopy=ops;
	vector<int> line1;
	vector<int> line2;
	for(int i=0;i<numbers.size();i++){
		line1.push_back(i);
	}
	for(int i=0;i<ops.size();i++){
		line2.push_back(i);
	}
	int* linear1=&line1[0];
	int* linear2=&line2[0];
	//making all permutaion of expressions and finding max
	int max=-100;
	do{
		do{
			int k=0,l=0;
			for(int i=0;i<numbers.size();i++){
				numbers[i]=numcopy[linear1[i]];
				
			}
			for(int i=0;i<ops.size();i++){
				ops[i]=opscopy[linear2[i]];
			}
			vector<char> expression(0);
			for(int i=0;i<oper.size();i++){
				if(!(i%2)){
					expression.push_back(numbers[k]);
					k++;
				}
				else{
					expression.push_back(ops[l]);
					l++;
				}
			}
			vector<char> infix(0);
			infix=infixtopostfix(expression);
			char cat;
			postfix ant;
			ant.calculate(infix);
			value=ant.show();
			if(max<value){
				max=value;
				resexp=expression;
			}
		}while(next_permutation(linear2,linear2+ops.size()));
	}while(next_permutation(linear1,linear1+numbers.size()));
	for(int i=0;i<resexp.size();i++){
		cout<<resexp[i];
	}
	cout<<"\n"<<max;
//getting cordinates of final expresiions
	for(int i=0;i<oper.size();i++){
		for(int j=0;j<oper.size();j++){              
		  	if (resexp[i]==oper[j]){
		    	output.push_back(loca[j]);
		        oper[j]='~';
		    }
		}              
	}
//bot position tracking
/*	int g=0;
	Mat feed1;
	while(1){
		cap>>feed;
		bothead.x=0;bothead.y=0;
		bottail.x=0;bottail.y=0;
		int n=0,m=0;
		//cvtColor(feed1,feed,CV_BGR2HSV);
		for(int i=0;i<feed.rows;i++){
			for(int j=0;j<feed.cols;j++){
				if(feed.at<Vec3b>(i,j)[0]<136 && feed.at<Vec3b>(i,j)[0]>116 && feed.at<Vec3b>(i,j)[1]>245 && feed.at<Vec3b>(i,j)[2]>238 && feed.at<Vec3b>(i,j)[2]<258){
					bothead.y+=i;
					bothead.x+=j;
					n++;
				}
				if(feed.at<Vec3b>(i,j)[0]<200 && feed.at<Vec3b>(i,j)[0]>179 && feed.at<Vec3b>(i,j)[1]>110 && feed.at<Vec3b>(i,j)[1]<135 && feed.at<Vec3b>(i,j)[2]>179 && feed.at<Vec3b>(i,j)[2]<199){
					bottail.y+=i;
					bottail.x+=j;
					m++;
				}
			}
		}
		if((bothead.y-bottail.y)>0){
			botangle=acos((bothead.x-bottail.x)/sqrt((bothead.x-bottail.x)*(bothead.x-bottail.x)+(bothead.y-bottail.y)*(bothead.y-bottail.y)));
		}
		else{
			botangle=3.14+acos((bothead.x-bottail.x)/sqrt((bothead.x-bottail.x)*(bothead.x-bottail.x)+(bothead.y-bottail.y)*(bothead.y-bottail.y)));
		}
		if((bothead.y-output[g].y)>0){
			charangle=acos((bothead.x-output[g].x)/sqrt((bothead.x-output[g].x)*(bothead.x-output[g].x)+(bothead.y-output[g].y)*(bothead.y-output[g].y)));
		}
		else{
			charangle=3.14+acos((bothead.x-output[g].x)/sqrt((bothead.x-output[g].x)*(bothead.x-output[g].x)+(bothead.y-output[g].y)*(bothead.y-output[g].y)));
		}
		if(charangle<3.14){
			if(botangle>charangle+3.14 || botangle<charangle){
				if(abs(botangle-charangle)>0.01745){
					sendCommand("a");  
				}
				else{
					sendCommand("w");
				}
			}
			else{ 
				if(abs(botangle-charangle)>0.01745){
					sendCommand("d");
				}
				else{
					sendCommand("w");
				}
			}	
		}

		else{
			if(botangle<charangle-3.14 && botangle>charangle){
				if(abs(botangle-charangle)>0.01745){
					sendCommand("d");
				}
				else{
					sendCommand("w");
				}
			}
			else{
				if(abs(botangle-charangle>0.01745)){
					sendCommand("a");
				}
				else{
					sendCommand("w");
				}
			}
		}
		if((bothead.x-output[g].x<3)&&(bothead.x-output[g].x<3)){
            sendCommand("l");
            g++;
 		}
	}*/
}



