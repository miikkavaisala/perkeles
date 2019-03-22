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
smiley_face(float* domain, int idx, float xx, float yy, float zz, float size, float radius1, float radius2)
{
    float rr    = sqrt(xx*xx + yy*yy + zz*zz); 
    float theta = acos(zz/rr) * 180.0/PI;
    float phi   = atan2(yy, xx) * 180.0/PI; 
    domain[idx] = 1.0;
    
    float rmax = (size/2.0);
    rr = rr/rmax;

    if (rr > radius1) {
        domain[idx] = 0.0;
    } else if (rr < radius2) {
        domain[idx] = 1.0;
    } else {
        if (theta > 30.0 && theta < 60.0) {
            if ((phi > 0.0  && phi < 30.0) || (phi > 60.0  && phi < 90.0)) {
                domain[idx] = 0.0;
            } else {
                domain[idx] = 1.0;
            }
        } else if ((theta > 120.0 && theta < 150.0)) {
            if (phi > 0.0  && phi < 90.0) {
                domain[idx] = 0.0;
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
coordinate(int* idx, float* xx, float* yy, float* zz, float* size, int i, int j, int k, int NX, float dx) 
{
    *size = float(NX)*dx;
    float origin = *size/2.0;
    *xx = float(i)*dx - origin;
    *yy = float(j)*dx - origin;
    *zz = float(k)*dx - origin;
    *idx = i + j*NX + k*NX*NX;
}

void 
init_domain(float* domain, int NX, float dx, float radius1, float radius2)
{
    int idx;
    float xx, yy, zz, size;

    //Make a smiley face


    for (int k=0; k < NX; k++ ) {
        for (int j=0; j < NX; j++ ) {
            for (int i=0; i < NX; i++ ) {
                 coordinate(&idx, &xx, &yy, &zz, &size, i, j, k, NX, dx);
                 smiley_face(domain, idx, xx, yy, zz, size, radius1, radius2);
            }       
        }       
    }       
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

    //   printf(" %f ", write_buf);

       fwrite(&write_buf, sizeof(float), 1, save_ptr);
   }
   printf(" \n ");
   fclose(save_ptr);
   
}

void 
swap_pointers(float** a, float** b)
{
	float* temp = *a;
	*a = *b;
	*b = temp;
}

// Integrator for the column density. The for loop is needed bacause the depth
// axis of integrated domain is not allocated to a thread index
__device__ void 
integrate_column_density(float* d_domain, float* d_image, int ind_pix, int NX, float dx)
{
    //Assume that the "detector" is initially empty
    d_image[ind_pix] = 0.0; 
    //OK printf("%i %f   ", NX, dx);
    //Integrate in depth. 
    for (int kk = 0; kk<NX; kk++) {
        int ind = ind_pix + kk*NX*NX;
        d_image[ind_pix] += dx*d_domain[ind];
        //printf("%f %f %i %i    ", d_image[ind_pix], d_domain[ind], ind_pix, ind);
    }
 //   printf(" END LOOP ");
 //   printf("DEVICE d_image %f ind_pix %i \n", d_image[ind_pix], ind_pix);
}


// CUDA kernel which organizes the map construction by allocating the correct
// indices and calling the integrator
__global__ void
make_map(float* d_domain, float* d_image, int NX, float dx) 
{
    int ii = threadIdx.x + blockIdx.x*blockDim.x;
    int jj = threadIdx.y + blockIdx.y*blockDim.y;
    int ind_pix = ii + jj*NX; 

    //printf("threadIdx.x %i blockIdx.x %i blockDim.x %i    ", threadIdx.x, blockIdx.x, blockDim.x);
    //printf("NX %i, dx %f, ii %i jj %i ind_pix %i    ", NX, dx, ii, jj, ind_pix);

    integrate_column_density(d_domain, d_image, ind_pix, NX, dx);
    
}
       
// The main() is run on the host, from which the CUDA kernels are called.
int 
main()
{
    float *h_domain, *d_domain;
    float *h_image,  *d_image;
    float dx = 1.0, radius1 = 0.9, radius2 = 0.6;
    int NX = 256;

    float max_rot = 360.0; //in deg
    float stride  = 60.0;    //deg 

    size_t domain_size = sizeof(float) * NX*NX*NX;
    size_t image_size  = sizeof(float) *    NX*NX;

    int NumThreads = 16; //MAXIMUM 32 (32^2 = 1024)
    int Blocks     = NX/NumThreads;
    dim3 threadsPerBlock(NumThreads, NumThreads);
    dim3 numBlocks(Blocks, Blocks);

    if (NX % NumThreads != 0) {
        printf("NX should div by NumThreads! Now %i / %i = %f \n", 
                NX, NumThreads, float(NX) / float(NumThreads));
        return 666;  
    }

    // Init density field in on HOST
    h_domain = (float*)malloc(domain_size);
    h_image  = (float*)malloc(image_size);
    // Make the smiley face
    init_domain(h_domain, NX, dx, radius1, radius2);

    //Allocate DEVICE memory
    cudaMalloc(&d_domain, domain_size);
    cudaMalloc(&d_image,  image_size);

    //Transfer domain (happy face) to HOST    
    cudaMemcpy(d_domain, h_domain, domain_size, cudaMemcpyHostToDevice);    

    float angle = 0.0; 
    while (angle <= max_rot) {
        //Calculate column density
        make_map<<<numBlocks, threadsPerBlock>>>(d_domain, d_image, NX, dx); 
   
        cudaDeviceSynchronize();

        //Make rotation

        cudaDeviceSynchronize();

        //swap_ptrs(&d_domain, &d_domain_buffer);

        //Send image to host memory
        cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);    

        write_map(h_image, NX, image_size, angle);

        //Rotate 
        angle += stride;
    }


    //Free allocated gpu memory
    cudaFree(d_domain); 
    cudaFree(d_image ); 

    free(h_domain);
    free(h_image );

    return 0; 
}







