// MIT License
// 
// Copyright (c) 2019 Miikka Väisälä
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.



#include <cstdio>
#include <cmath>

#define DEBUG 0 
#define PI 3.14159265

void 
happy_face(float* domain, int idx, float xx, float yy, float zz, float size, 
            float radius1, float radius2)
{
    float rr    = sqrt(xx*xx + yy*yy + zz*zz); 
    float theta = acos(xx/rr) * 180.0/PI;
    float phi   = atan2(yy, zz) * 180.0/PI + 45.0;

    domain[idx] = 1.0;
    
    float rmax = (size/2.0);
    rr = rr/rmax;

    if (rr > radius1) {
        domain[idx] = 0.0;
    } else if (rr < radius2) {
        domain[idx] = 1.0;
    } else {
        if (theta > 40.0 && theta < 80.0) {
            if ((phi > 0.0  && phi < 40.0) || (phi > 50.0  && phi < 90.0)) {
                domain[idx] = 10.0;
            } else {
                domain[idx] = 1.0;
            }
        } else if ((theta > 120.0 && theta < 150.0)) {
            if (phi > 0.0  && phi < 90.0) {
                domain[idx] = 5.0;
            } else {
                domain[idx] = 1.0;
            }
        } else {
            domain[idx] = 1.0;
        } 
    }

#if DEBUG == 1
    if (rr >= radius2 && rr <= radius1)
    printf("SMILE %.3e %.3e %.3e, %1.1f %1.1f, %4.1f %4.1f %4.1f, %1.1f \n", 
                  xx, yy, zz, radius1, radius2,  rr, theta, phi, domain[idx]);
#endif
}

void 
coordinate(int* idx, float* xx, float* yy, float* zz, float* size, 
           int i, int j, int k, int NX, float dx) 
{
    *size = float(NX)*dx;
    float origin = *size/2.0;
    *xx = float(i)*dx - origin;
    *yy = float(j)*dx - origin;
    *zz = float(k)*dx - origin;
    *idx = i + j*NX + k*NX*NX;
}

void 
init_domain(float* domain, float* domain_buffer, int NX, float dx, 
            float radius1, float radius2)
{
    int idx;
    float xx, yy, zz, size;

    //Make the Happy Face
    for (int k=0; k < NX; k++ ) {
        for (int j=0; j < NX; j++ ) {
            for (int i=0; i < NX; i++ ) {
                 coordinate(&idx, &xx, &yy, &zz, &size, i, j, k, NX, dx);
                 happy_face(domain, idx, xx, yy, zz, size, radius1, radius2);
                 domain_buffer[i + j*NX + k*NX*NX] = 0.0;

            }       
        } 
        printf("...%.1f percent", 100.0*float(NX + NX*NX + k*NX*NX)/float(NX*NX*NX));      
    }       
    printf("\n");
}

void 
write_map(float* h_image, int NX, size_t image_size, float angle)
{
   const size_t n = NX*NX;
   
   const char* fname = "happy_face";
   char cstep[10];
   char filename[80] = "\0";
   
   sprintf(cstep, "%d", int(angle));
   
   strcat(filename, fname);
   strcat(filename, "_");
   strcat(filename, cstep);
   strcat(filename, ".map");
   
   printf("Savefile %s \n", filename);
   
   FILE* save_ptr = fopen(filename,"wb");
   
   float write_buf =  (float) angle;
   fwrite(&write_buf, sizeof(float), 1, save_ptr);
   //Map data
   for (size_t i = 0; i < n; ++i) {
       const float point_val = h_image[i];
       float write_buf =  (float) point_val;

       fwrite(&write_buf, sizeof(float), 1, save_ptr);
   }
   printf(" \n ");
   fclose(save_ptr);
   
}



//
//
//
//////////////////////////////////////////////////////
// ALL GPU RELATED PART BELLOW THIS POINT ======>   //
// (Everything above is just auxuliary C code fluff)//
//////////////////////////////////////////////////////
//
//
//


// Swap pointers arround the recycle buffer
void 
swap_pointers(float** aa, float** bb)
{
	float* temp = *aa;
	*aa = *bb;
	*bb = temp;
}

//Tranfer values from the chosed cell. This completely lacks any intertolation,
//so it quite stupid. However, enough for the demo purposes for now. Will
//result in some noticeable graphical errors. 
__device__ int
get_ind_source_stupid(const int ii, const int jj, const int kk, const int NX, 
                      const float stride) 
{
    float radians = stride * PI/180.0;

    float x_tgt = (float(jj)-float(NX)/2)*cos(radians) 
                - (float(kk)-float(NX)/2)*sin(radians);
    float z_tgt = (float(jj)-float(NX)/2)*sin(radians) 
                + (float(kk)-float(NX)/2)*cos(radians);

    int jj_tgt = floor(x_tgt)+NX/2;
    int kk_tgt = floor(z_tgt)+NX/2;

    int ind_source;
    if (jj_tgt >= 0 && jj_tgt < NX && kk_tgt >= 0 && kk_tgt < NX) {
        ind_source = ii + jj_tgt*NX + kk_tgt*NX*NX; 
    } else {
        ind_source = -1.0;
    }

    return ind_source;
}


// Here we rotate the sphere using a GPU. A buffer array is used, because
// otherwise there will be problems with sychronizing the result.
__global__ void 
rotate(float* d_domain, float* d_domain_buffer, const int NX, const float dx, 
       const float stride)
{
    int ii = threadIdx.x + blockIdx.x*blockDim.x;
    int jj = threadIdx.y + blockIdx.y*blockDim.y;
    int kk = threadIdx.z + blockIdx.z*blockDim.z;
    int ind_target = ii + jj*NX + kk*NX*NX; 

    int ind_source;
    ind_source = get_ind_source_stupid(ii, jj, kk, NX, stride);

    if (ind_source >= 0) {
        d_domain_buffer[ind_target] = d_domain[ind_source];
    } else {
	// Zero if trying o access outside of domain bounds. 
        d_domain_buffer[ind_target] = 0.0;
    }

}

// Integrator for the column density. The for loop is needed bacause the depth
// axis of integrated domain is not allocated to a thread index
__device__ void 
integrate_column_density(float* d_domain, float* d_image, const  int ind_pix, 
const int NX, const float dx)
{
    //Assume that the "detector" is initially empty
    d_image[ind_pix] = 0.0; 
    //Integrate in depth. 
    for (int kk = 0; kk<NX; kk++) {
        int ind = ind_pix + kk*NX*NX;
        d_image[ind_pix] += dx*d_domain[ind];
    }
}


// CUDA kernel which organizes the map construction by allocating the correct
// indices and calling the integrator
__global__ void
make_map(float* d_domain, float* d_image, int NX, float dx) 
{
    int ii = threadIdx.x + blockIdx.x*blockDim.x;
    int kk = threadIdx.y + blockIdx.y*blockDim.y;
    int ind_pix = ii + kk*NX; 

    integrate_column_density(d_domain, d_image, ind_pix, NX, dx);
    
}

//////////////////////////////////////////////////////////////
//                                                          //
// DOWNLOAD FROM: https://github.com/miikkavaisala/perkeles //
//                                                          //
//////////////////////////////////////////////////////////////
    
// **BASICS OF GPU PROGRAMMING WITH CUDA**

// 1. Increasing demands of computation require exotic approaches. Graphics
//    processing units (GPUs) are a popular tool for accelerating computation.
//    --> A multi-GPU node can be really compact and affordable for its
//        computing power; assuming you write GPU compatible code (a big issue
//        for most users). 

// 2. Popular uses for GPUs:
//    * Graphics (e.g. almost all computer games.)
//    * Blockchain / Cryptocurrencies (e.g. Dogecoin)
//    * Machine learning (E.g. with Tensorflow)
//    * Radiative transfer. (E.g. this demo here, SOC)
//    * Astronomical data reduction/analysis (??? glCLEAN, FADRA, FATBOY)
//    * Fluid dynamics / MHD (E.g. Astaroth, Fargo3D, Gamer-2)

// 3. GPUs are made for computing a humongous number of floating point
//    operations in parallel.
//    --> Good tool when performance is bound by computation. Bad tool when
//        bound by memory. Some uses are more equal than others. 

// 4. Known API: 
//    * CUDA, by Nvidia (most popular, used here)
//    * OpenCL, by Khronos Group (open standard)
//    * OpenACC, by OpenACC.org (directive based, like OpenMP) 

// 5. Programming GPUs is not that difficult, if you are writing C code anyway.

// 6. Challenges: 
//    * Porting old things takes time - exitisting tool limited.
//    * Debugging is challenging.
//    * For demanding optimizations memory require special attention.


// CODE TIME! 

   
// The main() is run on the HOST, from which the CUDA kernels are called.      
int 
main()
{
    //Define pointers for HOST and DEVICE arrays. 
    float *h_domain, *d_domain;
    float *h_domain_buffer, *d_domain_buffer;
    float *h_image,  *d_image;

    //Domain settings
    float dx = 1.0, radius1 = 0.9, radius2 = 0.6;
    int NX = 512;

    //Rotation settings
    float max_rot = 360.0; //in deg
    float stride  = 1.0;    //deg 

    //Determine the required memory
    size_t domain_size = sizeof(float) * NX*NX*NX;
    size_t image_size  = sizeof(float) *    NX*NX;

    //Integrator thread and block settings
    int  RAD_NumThreads = 16; //MAXIMUM 32 (32^2 = 1024)
    int  RAD_Blocks     = NX/RAD_NumThreads;
    dim3 RAD_threadsPerBlock(RAD_NumThreads, RAD_NumThreads);
    dim3 RAD_numBlocks(RAD_Blocks, RAD_Blocks);

    //Rotatator thread and block settings
    int  ROT_NumThreads = 8; //MAXIMUM 8 (32^2 = 1024)
    int  ROT_Blocks     = NX/ROT_NumThreads;
    dim3 ROT_threadsPerBlock(ROT_NumThreads, ROT_NumThreads, ROT_NumThreads);
    dim3 ROT_numBlocks(ROT_Blocks, ROT_Blocks, ROT_Blocks);

    //To prevent mismatch with threads
    if (NX % RAD_NumThreads != 0) {
        printf("NX should div by RAD_NumThreads! Now %i / %i = %f \n", 
                NX, RAD_NumThreads, float(NX) / float(RAD_NumThreads));
        return 666;  
    }

    if (NX % ROT_NumThreads != 0) {
        printf("NX should div by ROT_NumThreads! Now %i / %i = %f \n", 
                NX, ROT_NumThreads, float(NX) / float(ROT_NumThreads));
        return 666;  
    }

    //Allocate HOST memory 
    h_domain        = (float*)malloc(domain_size);
    h_domain_buffer = (float*)malloc(domain_size);
    h_image         = (float*)malloc(image_size);

    // Init density field on HOST
    // Make the Happy Face
    printf("Making Happy Face... \n");
    init_domain(h_domain, h_domain_buffer, NX, dx, radius1, radius2);
    printf("Happy day! \n");

    //Allocate DEVICE memory
    cudaMalloc(&d_domain,        domain_size);
    cudaMalloc(&d_domain_buffer, domain_size);
    cudaMalloc(&d_image,          image_size);

    //Transfer domain (happy face) to HOST    
    cudaMemcpy(d_domain, h_domain, domain_size, cudaMemcpyHostToDevice);    
    cudaMemcpy(d_domain_buffer, h_domain_buffer, domain_size, cudaMemcpyHostToDevice);    

    float angle = 0.0; 
    while (angle <= max_rot) {
        //Calculate column density
        make_map<<<RAD_numBlocks, RAD_threadsPerBlock>>>(d_domain, d_image, NX, dx); 
	cudaDeviceSynchronize();

        //Send image to host memory
        cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);    
        cudaDeviceSynchronize();

        //Save result
        write_map(h_image, NX, image_size, angle);

        //Make rotation
        rotate<<<ROT_numBlocks, ROT_threadsPerBlock>>>(d_domain, d_domain_buffer, 
                                                       NX, dx, stride);
        cudaDeviceSynchronize();

        //Swat buffer pointer. 
        swap_pointers(&d_domain, &d_domain_buffer);

        angle += stride;
    }


    //Free allocated gpu DEVICE memory
    cudaFree(d_domain); 
    cudaFree(d_domain_buffer); 
    cudaFree(d_image ); 

    //Free HOST memory
    free(h_domain);
    free(h_image );

    return 0; 
}







