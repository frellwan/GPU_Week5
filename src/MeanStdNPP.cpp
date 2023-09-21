/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif


#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <dirent.h>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_statistics_functions.h>

#include <helper_cuda.h>
#include <helper_string.h>


// Save an 32bit floating point gray-scale image to disk.
void savePic(const std::string &rFileName, const npp::ImageCPU_32f_C1 &rImage)
{
  // create the result image storage using FreeImage so we can easily save to disk
  FIBITMAP *pResultBitmap = FreeImage_AllocateT(FIT_FLOAT, rImage.width(), rImage.height(), 8 /* bits per pixel */);
  NPP_ASSERT_NOT_NULL(pResultBitmap);
  unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
  Npp8u *pDstLine = FreeImage_GetBits(pResultBitmap);
  const Npp32f *pSrcLine = rImage.data();
  unsigned int nSrcPitch = rImage.pitch();

  // Copy the image data to the destination file
  memcpy(pDstLine, pSrcLine, (rImage.width() * sizeof(Npp32f) * rImage.height()));

  // now save the result image
  bool bSuccess;
  bSuccess = FreeImage_Save(FIF_TIFF, pResultBitmap, rFileName.c_str(), 0) == TRUE;
  NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
  FreeImage_Unload(pResultBitmap);
}

// Print The NPP information to the screen
bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  std::cout << "NPP Library Version " << libVer->major << '.' << libVer->minor << '.' << libVer->build << std::endl;

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  std::cout << "  CUDA Driver  Version: " << driverVersion / 1000 << '.' << (driverVersion % 100) / 10 << std::endl;
  std::cout << "  CUDA Runtime Version:  " << runtimeVersion / 1000 << '.' << (runtimeVersion % 100) / 10 << std::endl;

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}


// Read in MNIST data set files and convert to a standardized 32 bit float version
int main(int argc, char *argv[])
{
  std::cout << argv[0] << " Starting..." << std::endl << std::endl;

  DIR *dir;
  struct dirent *ent;
  std::vector<std::string> filenames;
  std::stringstream filename;

  if ((dir = opendir(".//img")) != NULL)
  {
    while((ent = readdir(dir)) != NULL)
    {
      if (ent->d_type == DT_REG)
      {                                    // if entry is a regular file
        std::string fname = ent->d_name; // filename
        
        // if filename's last characters are extension .jpg
        if (fname.find(".jpg", (fname.length() - 4)) != std::string::npos)
        {
          filenames.push_back(ent->d_name);
        }
      }
    }
    closedir(dir);
  }
  else
  {
    perror("");
  }

  findCudaDevice(argc, (const char **)argv);

  // Make sure that NPP is available
  if (printfNPPinfo(argc, argv) == false)
  {
    exit(EXIT_SUCCESS);
  }

  std::string logFileName;
  logFileName = ".//log";
  std::ofstream logfile;
  logfile.open(logFileName);

  for (int i = 0; i < filenames.size(); i++)
  {
    try
    {
      std::string sFilename;
      char *filePath;

      if (checkCmdLineFlag(argc, (const char **)argv, "input"))
      {
        getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
      }
      else
      {
        // Clear stringstream each time
        filename.str(std::string());
        filename << ".//img//" << filenames[i];
        sFilename = filename.str();
        filePath = sdkFindFilePath(sFilename.c_str(), argv[0]);
      }

      // If the file was found in the filepath
      if (filePath)
      {
        sFilename = filePath;
      }

      // Else just open this file in the current directory
      else
      {
        sFilename = "Lena.pgm";
      }

      // if we specify the filename at the command line, then we only test
      // sFilename[0].
      int file_errors = 0;
      std::ifstream infile(sFilename.data(), std::ifstream::in);

      if (infile.good())
      {
        logfile << "MeanStdNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
      }
      else
      {
        std::cout << "MeanStdNPP unable to open: <" << sFilename.data() << ">"
                  << std::endl;
        logfile << "MeanStdNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
        file_errors++;
        infile.close();
      }

      if (file_errors > 0)
      {
        exit(EXIT_FAILURE);
      }

      std::string sResultFilename = ".//std_img//" + filenames[i];

      std::string::size_type dot = sResultFilename.rfind('.');

      if (dot != std::string::npos)
      {
        sResultFilename = sResultFilename.substr(0, dot);
      }

      sResultFilename += "_meanstd.tif";

      if (checkCmdLineFlag(argc, (const char **)argv, "output"))
      {
        char *outputFilePath;
        getCmdLineArgumentString(argc, (const char **)argv, "output",
                                 &outputFilePath);
        sResultFilename = outputFilePath;
      }

      // declare a host image object for an 8-bit grayscale image
      npp::ImageCPU_8u_C1 oHostSrc;

      // load gray-scale image from disk
      npp::loadImage(sFilename, oHostSrc);

      // declare a device image and copy construct from the host image,
      // i.e. upload host to device
      npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
      npp::ImageNPP_32f_C1 Device_Src_32f(oDeviceSrc.size());

      NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
      NppiSize oSrcSize32f = {(int)Device_Src_32f.width(), (int)Device_Src_32f.height()};
      NppiPoint oSrcOffset = {0, 0};

      // create struct with ROI size
      NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
      NppiSize oSizeROI32f = {(int)Device_Src_32f.width(), (int)Device_Src_32f.height()};

      // allocate device image of appropriately reduced size
      npp::ImageNPP_32f_C1 oScratchDev(oSizeROI32f.width, oSizeROI32f.height);
      npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI32f.width, oSizeROI32f.height);

      // define variables for mean and stdandard deviation for both host and device and allocate device memory
      double *mean_dev, *stddev_dev;
      double mean, stddev;

      // Allocate the memory on the GPU
      cudaMalloc((void **)&mean_dev, sizeof(double));
      cudaMalloc((void **)&stddev_dev, sizeof(double));

      // Compute the appropriate size of the scratch-memory buffer
      int nBufferSizeMeanStd;
      nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &nBufferSizeMeanStd);

      // Allocate the scratch buffer
      Npp8u *pDeviceBuffer;
      checkCudaErrors(cudaMalloc((void **)(&pDeviceBuffer), nBufferSizeMeanStd));

      // Convert image from 8u to 32f for calculations
      NPP_CHECK_NPP(nppiConvert_8u32f_C1R(oDeviceSrc.data(), (int)oDeviceSrc.width(), Device_Src_32f.data(), (int)Device_Src_32f.width() * 4, oSizeROI));

      // Calculate and print the mean and standard deviation of the pixels in the image using the scratch-memory buffer allocated above
      NPP_CHECK_NPP(nppiMean_StdDev_32f_C1R(Device_Src_32f.data(), (int)Device_Src_32f.width() * 4, oSrcSize32f, pDeviceBuffer, mean_dev, stddev_dev));
      checkCudaErrors(cudaMemcpy(&mean, mean_dev, sizeof(double), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(&stddev, stddev_dev, sizeof(double), cudaMemcpyDeviceToHost));

      // Log the original mean and stddev of the image
      logfile << std::fixed << std::setprecision(5);
      logfile << "Mean Orig: " << mean << "     "
              << "StdDev Orig: " << stddev << '\n';

      // Mean Center the image - subtract the image mean pixel value from each pixel
      nppiSubC_32f_C1R(Device_Src_32f.data(), (int)Device_Src_32f.width() * 4, mean, oScratchDev.data(), (int)Device_Src_32f.width() * 4, oSrcSize32f);

      // Standardize the mean centered image - divide each pixel value by the image standard deviation
      nppiDivC_32f_C1R(oScratchDev.data(), (int)oScratchDev.width() * 4, stddev, oScratchDev.data(), (int)Device_Src_32f.width() * 4, oSrcSize32f);

      // Calculate and print the mean and standard deviation of the mean centered and standardized image
      NPP_CHECK_NPP(nppiMean_StdDev_32f_C1R(oScratchDev.data(), (int)oScratchDev.width() * 4, oSrcSize32f, pDeviceBuffer, mean_dev, stddev_dev));
      checkCudaErrors(cudaMemcpy(&mean, mean_dev, sizeof(double), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(&stddev, stddev_dev, sizeof(double), cudaMemcpyDeviceToHost));

      // Log the new mean and stddev of the image after the standardization process
      logfile << "Mean std: " << mean << "     "
              << "StdDev std: " << stddev << '\n';

      // declare a host image for the result
      npp::ImageCPU_32f_C1 oHostDst(oScratchDev.size());
 
      // and copy the device result data into it
      oScratchDev.copyTo(oHostDst.data(), oHostDst.pitch());

      // Save the image to the disc
      savePic(sResultFilename, oHostDst);
      logfile << "Saved image: " << sResultFilename << std::endl;

      // Free allocated memory after all images converted
      if (i == filenames.size()-1)
      {
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
        nppiFree(oScratchDev.data());
        nppiFree(Device_Src_32f.data());
      }

      cudaFree(mean_dev);
      cudaFree(stddev_dev);
      cudaFree(pDeviceBuffer);
    }
    catch (npp::Exception &rException)
    {
      std::cerr << "Program error! The following exception occurred: \n";
      std::cerr << rException << std::endl;
      std::cerr << "Aborting." << std::endl;

      exit(EXIT_FAILURE);
    }
    catch (...)
    {
      std::cerr << "Program error! An unknow type of exception occurred. \n";
      std::cerr << "Aborting." << std::endl;

      exit(EXIT_FAILURE);
      return -1;
    }
    cudaDeviceSynchronize();
  }
  logfile.close();

  return 0;
}
