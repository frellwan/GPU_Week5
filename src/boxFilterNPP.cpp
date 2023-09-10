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

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_statistics_functions.h>

#include <helper_cuda.h>
#include <helper_string.h>

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

int main(int argc, char *argv[])
{
  std::cout << argv[0] << " Starting..." << std::endl << std::endl;

  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(".//")) != NULL)
  {
    while((ent = readdir(dir)) != NULL)
    {
      std::cout << ent->d_name << '\n';
    }
    closedir(dir);
  }
  else
  {
    perror("");
  }

  try
  {
    //const std::filesystem::path sandbox{"GPU_Week5"};
    //for (const auto &dirEntry : std::filesystem::directory_entry("."))
    //  std::cout << dirEntry << std::endl;

    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    // Make sure that NPP is available
    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
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
      std::cout << "boxFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_boxFilter.pgm";

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
    // npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    npp::ImageNPP_32f_C1 oScratchDev(oSizeROI32f.width, oSizeROI32f.height);
    npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI32f.width, oSizeROI32f.height);

    // define variables for mean and stdandard deviation for both host and device and allocate device memory
    double *mean_dev, *stddev_dev;
    double mean, stddev;

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
    NPP_CHECK_NPP(nppiMean_StdDev_32f_C1R(Device_Src_32f.data(), (int)Device_Src_32f.width()*4, oSrcSize32f, pDeviceBuffer, mean_dev, stddev_dev));
    checkCudaErrors(cudaMemcpy(&mean, mean_dev, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&stddev, stddev_dev, sizeof(double), cudaMemcpyDeviceToHost));
    printf("Mean: %lf      StdDev: %lf \n", mean, stddev);

    // Mean Center the image - subtract the image mean pixel value from each pixel
    nppiSubC_32f_C1R(Device_Src_32f.data(), (int)Device_Src_32f.width() * 4, mean, oScratchDev.data(), (int)Device_Src_32f.width() * 4, oSrcSize32f);

    // Standardize the mean centered image - divide each pixel value by the image standard deviation
    nppiDivC_32f_C1R(oScratchDev.data(), (int)oScratchDev.width() * 4, stddev, oScratchDev.data(), (int)Device_Src_32f.width() * 4, oSrcSize32f);

    // Calculate and print the mean and standard deviation of the mean centered and standardized image
    NPP_CHECK_NPP(nppiMean_StdDev_32f_C1R(oScratchDev.data(), (int)oScratchDev.width() * 4, oSrcSize32f, pDeviceBuffer, mean_dev, stddev_dev));
    checkCudaErrors(cudaMemcpy(&mean, mean_dev, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&stddev, stddev_dev, sizeof(double), cudaMemcpyDeviceToHost));
    printf("Mean: %lf      StdDev: %lf \n", mean, stddev);

    // declare a host image for the result
    npp::ImageCPU_32f_C1 oHostDst(oScratchDev.size());
    // and copy the device result data into it
    oScratchDev.copyTo(oHostDst.data(), oHostDst.pitch());

    //saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    // Free allocated memory
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
    nppiFree(oScratchDev.data());
    nppiFree(Device_Src_32f.data());

    cudaFree(mean_dev);
    cudaFree(stddev_dev);
    cudaFree(&pDeviceBuffer);

    exit(EXIT_SUCCESS);
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

  return 0;
}
