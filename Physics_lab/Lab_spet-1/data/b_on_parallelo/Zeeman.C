//========================================================================================================
//
//  macro di ROOT per fittare il segnale dell'esperienza di Zeeman del laboratorio
//  di Fisica del terzo anno dell'Università di Padova
//
//  Zeeman(const char* infile, bool fit = false, int lpix = 0, int upix = 0)
//
//  parametri in ingresso:
//    - infile: nome del file .csv da analizzare
//    - fit:    se true fa il fit di una distribuzione gaussiana(*)
//    - lpix:   il limite inferiore da considerare per il fit
//    - upix:   il limite superiore da considerare per il fit
//
//    se i limiti per il fit (lpix, upix) sono entrambi nulli cerca il picco di intensità massima e 
//    prova a fare il fit di quello
//
//    una volta eseguita la macro è disponibile l'istogramma "zeeman" che può essere utilizzato per
//    calcolare fit al di fuori della macro o compiere qualunque azione si voglia sull'istogramma
//    (ovviamente permessa dall'istogramma stesso)
//
//  Prof. R. Stroili (2024)
//
// (*) per permettere al fit di adattarsi meglio alle distorsioni dei picchi di interferenza è stata
// introdotta una piccola deformazione lineare della larghezza.
//
//========================================================================================================

#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include "TROOT.h"
#include "TGraph.h"
#include "TF1.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "TStyle.h"

//
//  read data extracted from the Fiji software
//
TH1F* readData(const char* infile, vector<double>& x, vector<double>& y) {

  //  open the input stream
  std::ifstream ifile(infile);

  // check stream status
  if (!ifile) {
    std::cerr << "Can't open input file " << infile << " !!!" << std::endl;
    exit(-1);
  }

  // string used for data reading
  std::string line;

  // flag to skip the file header
  bool skip = true;
  while (ifile) {
    std::getline(ifile, line);
    if (not skip and line.size() > 1) {
      // remove the csv data separators (, and  ;)
      std::replace(line.begin(), line.end(), ',', ' ');
      std::replace(line.begin(), line.end(), ';', ' ');
      // instanciate a stringstream to extract the values
      std::stringstream ss(line);
      float v1, v2;
      ss >> v1 >> v2;
      // put time and amplitude values in their vectors
      x.push_back(v1);
      // invert the signal: work with positive signals
      y.push_back(v2);
    }
    // check if we have reached the bottom of the file header, delimited by the "TIME" word
    if (line.find("Distance_") != std::string::npos) skip = false;
  }
  ifile.close();
  if (x.size() == 0) {
    std::cerr << "empty or wrong file: " << infile << "\n\t...\t exiting" << std::endl;
    exit(-1);
  }
  TH1F* h = 0;
  gROOT->GetObject("zeeman", h);
  if (h) delete h;
  h = new TH1F("zeeman", "Zeeman", x.size(), x[0], x[x.size() - 1]);
  h->SetContent(&y[0]);

  //  return the histogram generated with the input data
  return h;
}

//========================================================================================================

//
//  signal fitting function: gaussian + decaying exponential
//
Double_t myfunction(Double_t* x, Double_t* par) {
  // get the x value 
  Double_t xx = x[0];
  // temporary variable
  Double_t val = 0;
  // gaussian component with variablenon constant width
  double sigma =  par[2] + par[3]*xx;
  val = par[0] * exp(-0.5 * ((xx - par[1]) / sigma) * ((xx - par[1]) / sigma));

  //return the function value at the given x
  return val;
}

//========================================================================================================

//
//  fit the signal
//
void fitSignal(TH1F* histo, const vector<double>& x, const vector<double>& y, int lpix, int upix) {
  std::cout << lpix << " " << upix << std::endl;

  //   find the initial values for the function parameters

  // find the position of the maximum
  auto it = max_element(std::begin(y), std::end(y));
  int pos = it - y.begin();

  // find the "width" of the gaussian
  int llimit = 0;
  int ulimit = 0;
  float center = (upix + lpix) * 0.5;
  if (lpix == 0 and upix == 0) {
    for (int k = pos; k > 0; --k) {
      if (y[k] < (y[pos] * 0.7)) {
        llimit = k;
        break;;
      }
    }

    for (unsigned int k = pos; k < y.size(); ++k) {
      if (y[k] < (y[pos] * 0.7)) {
        ulimit = k;
        break;;
      }
    }
    int deltaH = (ulimit - llimit) * 0.5;
    lpix = x[pos - deltaH];
    upix = x[pos + deltaH];
    center = x[pos];
  }
  std::cout << lpix << " " << upix << std::endl;

  // lower time limit for fitting 
  float xllimit = 0.;

  // create the fitting function object
  TF1* f1 = new TF1("myfunc", myfunction, lpix, upix, 4);
  f1->SetNpx(2000);
  // set initial parameters and limits 
  f1->SetParameter(0, 20);
  f1->SetParLimits(0, 0, 10000);
  f1->SetParName(0, "GaussNorm");
  f1->SetParameter(1, center);
  f1->SetParLimits(1, center * 0.5, center * 1.5);
  f1->SetParName(1, "peak");
  f1->SetParameter(2, (upix - lpix) * 0.3);
  f1->SetParLimits(2, (upix - lpix) * 0.2, (upix - lpix) * 1.5);
  f1->SetParName(2, "width");
  f1->SetParameter(3, 0);
  f1->SetParLimits(3, -0.3, 0.3);
  f1->SetParName(3, "width_linear_deformation");

  // fit the signal
  histo->Fit("myfunc", "0+", "", lpix, upix);

}

//========================================================================================================

// 
//  main function
//
void Zeeman(const char* infile, bool fit = false, int lpix = 0, int upix = 0) {
  // create a null pointer to a TCanvas object
  TCanvas* can = 0;
  // look for a TCanvas named "Zeeman"
  auto list = gROOT->GetListOfCanvases();
  for (int k = 0; k < list->GetLast()+1; ++k) {
    can = (TCanvas*)list->At(k);
    // delete the canvas if it exists and create a new one
    if (strcmp(can->GetName(), "ZeemanCan")==0) delete can;
  }
  //  delete function "myfunc" if present
  TF1* f1 = 0;
  auto funl = gROOT->GetListOfFunctions();
  f1 = (TF1*)funl->FindObject("myfunc");
  if (f1) delete f1;

  // instantiate the containers
  std::vector<double> x;
  std::vector<double> y;

  // gStyle->SetTitleSize(.1, "Y");
  gStyle->SetLabelSize(.05, "XY");
  gStyle->SetStripDecimals(kFALSE);

  // read the data
  TH1F* hsig = readData(infile, x, y);

  // format the histogram
  hsig->SetTitle("zeeman;pixel;I");
  hsig->GetXaxis()->SetTitleSize(0.05);
  hsig->GetXaxis()->SetTitleOffset(1);
  hsig->GetYaxis()->SetTitleSize(0.05);
  hsig->GetYaxis()->SetTitleOffset(1);

  // create new TCanvas object
  can = new TCanvas("ZeemanCan", "Zeeman");
  // fit the signal
  if (fit) {
    fitSignal(hsig, x, y, lpix, upix);
  }

  // draw the signal with superimposed the fitted function
  hsig->Draw("");
  TF1* myf = 0; 
  // gROOT->GetObject("myfunc", myf);
  auto fl = gROOT->GetListOfFunctions();
  myf = (TF1*)fl->FindObject("myfunc");
  if (myf) {
    myf->Draw("same");
  }
}
