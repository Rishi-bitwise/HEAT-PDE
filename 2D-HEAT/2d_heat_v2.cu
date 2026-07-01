#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>

/*We have a tile of 16*16 threads. so 256 threads in a block. But how much shared memory can it have, that decides the halo around the 
actual tile, thereby giving us the number of time steps a specific kernel launch can calculate. In 1D, it was learnt that we need to launch 
multiple kernels so as to sync the data when we write to global memory, as long as we work with shared memory, we can calculate upto the h2 mark.
Shared memory limit is 32KB per block. We can increase to 48/64KB, but then we wont be able to launch 2 blocks per SM. SO which is better 32KB with
2 blocks or 48KB with 1 block. lets calculate then:

1)  32KB means 8K floats per block. our 256 threaded tile/block takes up 1KB, So we can have 25 such tiles for 1 block costing us (5*16+1)*(5*16+1)*4B
    of shared memory, = 26.3KB , shbx and shby the boundary values of shmem take up 2*[5*4]*16 *4B = 2.5KB , which easily fits into 32KB -26.3KB = 5.7KB.
2)  48KB means 12K floats per block. our 256 threaded block needs 1KB, We cannot get 49 tiles, total as it would cost 49KB. so its still 25 same as before,
    but with lesser SM-block occupancy.

We could try TILESIZE 8, shared memory per block would allow us 128 tiles, we could use 121 tiles. SHLOOPDIM=11. So blocks would be 64 threads,
SM can handle 2048, so each SM could hold 2K/64=32 blocks. But Sm has total 64KB of shared memory, so instead of 32 blocks it would hold,
2 blocks only, since we are trying to utilize 32KB max per block. But shbx and shby dont fit. These are the boundaries of tiles, to track the old values 
for prevx, and prevy. total boundary values will be 2*(11*10)*8 *4B = 7.4KB . our 121 shmem tiles took 121*8*8*4B = 30KB(without padding). 30+7.4 >32, so we 
cannot fit 11x11 .
But what about 9x9. (9*8+1)*(9*8+1) floats in shmem = 22KB. Boundary values need 2*[9*8]*8 *4B = 4.6KB . 4.6+21<32KB. so we can fit the 8x8 tiles in a 
9x9 grid of our shared memory. HALO2 will be 4 in this case. 4* 8*8 = 256. so we can have 256 time steps for 8x8 tiles. 


[BUG FIX-2] We have a problem, with shared memory. Again. We cannot update tile by tile, since each tile needs its adjacent tiles data to update
itself. We cannot write the tile into memory right away, before the next tile can readthe values. or can we. we can just store theboundary
values of the tiles. Say we have [t0, t1,t2], [t3,t4,t5], [t6,t7,t8]. we should store boundary values of t0 for t1 and t3. right boundary for 
t1 and bottom boundary for t3. So totally we have 12 such boundaries[in this example]. each boundary is 16. So we need 12*16 *4bytes = 2560 bytes.
Remember that we are going t0->t1->t2->t3->t4.... So t0 gets to have the old values of t1, but not the other way around, so we only need t0's 
old values to update t1. similarly for the rest. This temporal direction helps us since we now dont have to save t1's left boundary for t0. 
*/

#define SIDE 256                           //Actual dimension of the plate
#define TOTAL_TIME 1024*32                  //we will be skipping every 32 steps, from the kernel, but also we will record less to keep filesize small
#define TILESIZE 16
#define SHLOOPDIM 5                         //NUMBER of tiles in sharedmem, NEEDS TO BE ODD.
#define HALO2 (SHLOOPDIM/2)*TILESIZE                      //32 halos above, below left and right of the central tile
#define KERNEL_TIME_SKIPS 16
#define TOTAL_TIME_ROWS ((TOTAL_TIME)/(HALO2*KERNEL_TIME_SKIPS))
#define SHMEMDIM (SHLOOPDIM*TILESIZE)
#define PI 3.14159f
#define BOUNDARY_TEMP 10.0f
#define R 0.125f

__global__ void init_temp(float *d_in)
{
    int xid = threadIdx.x + blockIdx.x*TILESIZE;
    int yid = threadIdx.y + blockIdx.y*TILESIZE;

    d_in[yid*SIDE + xid] = 100*sinf(2*PI*(((float)(xid)+2*yid)/SIDE));
}

__global__ void run_sim(float *d_in, float *d_out)
{
    int xid = threadIdx.x + blockIdx.x*TILESIZE;
    int yid = threadIdx.y + blockIdx.y*TILESIZE;

    int thx = threadIdx.x;
    int thy = threadIdx.y;

    __shared__ float shmem[SHMEMDIM+1][SHMEMDIM+1];                 // Padding is necessary because we need the columns as well.
    //SHMEMDIM is the 3x3 tile with halos.
    __shared__ float shbx[SHLOOPDIM][SHLOOPDIM-1][TILESIZE];           //this is for boundary between t0/t1 or t4/t5. 
    __shared__ float shby[SHLOOPDIM-1][SHLOOPDIM][TILESIZE];           
    //this is for boundary between t0/t3 or t5/t8. Note that the dimensions are transposed, because we wanna follow t0->t1->t2->t3. 
    // that means that  


    //--------------------------------------LOAD THE DATA FROM GMEM INTO SHMEM--------------------------------------//

    int globalxstart = blockIdx.x*TILESIZE - HALO2;
    int globalystart = blockIdx.y*TILESIZE - HALO2;

    for(int i=0 ;i<SHLOOPDIM; i++)                      //SHLOOPDIM = 5/9
    {
        for(int j=0; j<SHLOOPDIM; j++)
        {
            int globalx = (globalxstart+j*TILESIZE+thx);
            int globaly = (globalystart+i*TILESIZE+thy);

            if (globalx<0 || globalx>=SIDE || globaly<0 || globaly>= SIDE)
                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = BOUNDARY_TEMP;
            else
                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = d_in[globaly*SIDE + globalx]; 
        }
    }

    __syncthreads();

    //--------------------------------------START THE UPDATE LOOP TO GO OVER THE TILES IN THE SHMEM--------------------------------------//

    for(int time=0; time<HALO2; time++)
    {
        //--------------------------------------LOAD THE DATA FROM SHMEM INTO SHBOUNDS--------------------------------------//

        /*
        shbx records : t0-t1, t1-t2, t3-t4, t6-t7, t7-t8. that is t0's last col, t1's last col, t3's last col, and so on. 
        It is not that important to get the best performance, we could just manually loop through them and store them using only 1 col pf 32 threads.
        But we can take 2 col of threads, point them at t0-end and t1-end and then increment by 3 tiles for 3 iterations to get the data. 
        */

        __syncthreads();
        if (thy<SHLOOPDIM-1)           //grab two rows of 32/16 threads. conditional branching penalty for thy only. 
        {
            for(int i=0; i<SHLOOPDIM; i++)
            {
                shbx[i][thy][thx] = shmem[i*TILESIZE + thx][thy*TILESIZE + TILESIZE-1];
                shby[thy][i][thx] = shmem[thy*TILESIZE + TILESIZE-1][i*TILESIZE+thx];           //Packing this here was good move. instead of another for/if
            }
        }
        __syncthreads();

        //--------------------------------------SHBOUNDS LOADED--------------------------------------//


        float prevx = 0;
        float prevy = 0;

        for(int i=0; i<SHLOOPDIM; i++)
        {

            for(int j=0; j<SHLOOPDIM; j++)
            {
                float curr = shmem[i*TILESIZE+thy][j*TILESIZE+thx];

                bool nextx_exists = (globalxstart + j*TILESIZE+thx+1)<SIDE;         //these are checks for the global memory.
                bool nexty_exists = (globalystart + i*TILESIZE+thy+1)<SIDE;
                bool prevx_exists = (globalxstart + j*TILESIZE+thx-1)>=0;
                bool prevy_exists = (globalystart + i*TILESIZE+thy-1)>=0;

                nextx_exists = nextx_exists && !(j==SHLOOPDIM-1 && thx==TILESIZE-1);                  //these are checks for shared memory
                nexty_exists = nexty_exists && !(i==SHLOOPDIM-1 && thy==TILESIZE-1);
                prevx_exists = prevx_exists && !(j==0 && thx==0);
                prevy_exists = prevy_exists && !(i==0 && thy==0);

                float nextx, nexty;
                if (nextx_exists)
                    nextx = shmem[i*TILESIZE+thy][j*TILESIZE+thx+1];       //*((int)(nextx_exists)) + BOUNDARY_TEMP*((int)(!nextx_exists));
                else
                    nextx = BOUNDARY_TEMP;

                if (nexty_exists)
                    nexty = shmem[i*TILESIZE+thy+1][j*TILESIZE+thx];       //*((int)(nexty_exists)) + BOUNDARY_TEMP*((int)(!nexty_exists));
                else
                    nexty = BOUNDARY_TEMP;

                //Unfortunately this is the [BUG FIX-1]. The previous tiles have already been updated. this is where shbx and shby shine!!!
                if (thx==0)                                 // HAS to be thx, because of the location within the tile 
                {   
                    if (j!=0)
                    {
                        if (prevx_exists)
                            prevx = shbx[i][j-1][thy]; 
                        else
                            prevx = BOUNDARY_TEMP;
                    }
                }
                else 
                    prevx = shmem[i*TILESIZE+thy][j*TILESIZE+thx-1]*((int)(prevx_exists)) + BOUNDARY_TEMP*((int)(!prevx_exists));

                if (thy==0)                                         //can be written with bool, but readability takes a hit.
                {
                    if (i!=0)
                    {
                        if (prevy_exists)
                            prevy = shby[i-1][j][thx];
                        else
                            prevy = BOUNDARY_TEMP;

                    }
                }
                else
                    prevy = shmem[i*TILESIZE+thy-1][j*TILESIZE+thx]*((int)(prevy_exists)) + BOUNDARY_TEMP*((int)(!prevy_exists));
                    
                float new_curr = curr + R*(nextx + nexty + prevx + prevy - 4*curr);

                __syncthreads();

                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = new_curr;

                __syncthreads();
            }
        }
    }

    d_out[yid*SIDE + xid] = shmem[HALO2+thy][HALO2+thx];
}

int main()
{
    float *h_arr;                           // It needs to record the 2d plate(SIDE*SIDE) for TOTAL_TIME_ROWS times
    int platedim = SIDE*SIDE;
    h_arr = (float*) malloc(platedim*TOTAL_TIME_ROWS*sizeof(float));

    float *d_in ,*d_out;
    float * tempo;
    cudaMalloc(&d_in, platedim*sizeof(float));
    cudaMalloc(&d_out, platedim*sizeof(float));

    int gridlen = (SIDE+TILESIZE-1)/TILESIZE;

    dim3 blocksize(TILESIZE, TILESIZE);

    dim3 gridsize(gridlen, gridlen);

    init_temp<<<gridsize, blocksize>>>(d_in);

    for(int i=0; i<TOTAL_TIME_ROWS; i++)
    {
        for(int j=0; j<KERNEL_TIME_SKIPS; j++)
        {
            run_sim<<<gridsize, blocksize>>>(d_in, d_out);
            tempo = d_in;
            d_in = d_out;
            d_out = tempo;
            //cudaMemcpy(d_in, d_out, platedim*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(&h_arr[i*platedim], d_in, platedim*sizeof(float), cudaMemcpyDeviceToHost);           //d_in now actually has the output since we switched the pointers
    }

    FILE *f;
    f = fopen("2d_heat.csv", "w");
    for(int i=0; i<TOTAL_TIME_ROWS; i++)
    {   
        for(int j=0;j<platedim; j++)
        {
            fprintf(f, "%.2f,", h_arr[i*platedim+j]);
        }
        fputs("10\n", f);
    }
    fclose(f);


    free(h_arr);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
