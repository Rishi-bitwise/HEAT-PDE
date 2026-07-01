%%cuda

//THIS IS FOR COLAB/SMALLER SYSTEMS THAT DONT NEED SUCH TIME GRANULARITY IN THE OUTPUT, JUST A ROUGH EVOLUTION. THIS IS V2.

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>


#define R 0.125f                        // This is the constant for the equation in question
#define ROD_LENGTH 128
#define BOUNDARY_TEMP 10                // Using dirichlet condition.
#define TILESIZE 1024                   //Same as blockDim
#define TOTAL_BLOCKS 16
#define TOTAL_TIME 1024*1024
#define TOTAL_PARTS TOTAL_BLOCKS*TILESIZE
#define TOTAL_HALOS 1024
#define TEMP_SCALING_FACTOR 676767.0f
#define PI 3.1415f
#define TOTAL_TIME_DATAPOINT_ROWS (TOTAL_TIME/(TOTAL_HALOS/2))


/* Here total number of threads is 1024*16 so 16k threads. each thread will handle one element of the rod. So the rod is divided into 16k parts.

Initially the rod will have a heat profile that is simply !NOT! parabolic. Not too complicated. From there we track the temperature evolution with time.
The update equation is u[n,t+1] = u[n,t] + R*(u[n-1,t] + u[n+1,t] - 2*u[n,t])

So the rod will have temperature with cols as elements and rows as time.

Temporal blocking is being used here [basically load more halos and slash the valid region per time step.]

Every kernel launch launches all the threads in question, and blocks. Each block loads into shared memory halos from adjacent blocks as well.
We can use these halos to time step within a single kernel launch and see through several time steps. this reduces number of kernel launches.
How do we use halos - say we have 'hnum' halos to left and right of the shared memory. and we have 1024 threads in our block.
2 Cases are possible : Middle block, or edge block.
For middle block, we have real data on either side which can be easily loaded into shared memory. the first 'hnum' number of threads of a block deal with the halos
So our halos are 1024, so 512 on either side. that means that first 512 threads load the data into the halos. now we apply the update, and ignore the start/last halos
, this update gives us the next values of our timestep. for every time step we ignore the first and last of the new shared memory temps, until we run out of the halos
so 512 on either side are garbage/ignored after 512 steps. At each time step, we record the middle 1024 values in the shared memory, and write to global memory.
Once all the blocks finish this, we copy this to main memory.

For Edge block its simple we just set the out of bounds values to the BOUNDARY_TEMP
*/


 __global__ void initialize_temp(float *U)
 {
    int xid = threadIdx.x + blockIdx.x*TILESIZE;
    //U[xid] = (-xid*xid + TOTAL_PARTS*(float)(xid))/TEMP_SCALING_FACTOR;
    U[xid] = 1000*sinf(2*PI*xid/1024.0f);
}



__global__ void Heat_PDE_1D(float *U)                                     //DIMENSIONS(U) = (1+1)*TOTAL_PARTS ; +1 is for the initial state of the rod for this kernel, the other is to store the output row.
{
    int xid = threadIdx.x + blockIdx.x*TILESIZE;

    /*if (xid>=TILESIZE*TOTAL_BLOCKS)                                     //TODO: MAKE THIS MORE ROBUST. (__syncthreads will cause an issue.)
    {
        return;
    }*/

    __shared__ float sharedtemp[TILESIZE + TOTAL_HALOS];                        //THis means 512 before sharedtemp[0] and after sharedtemp[-1]
    // Here the entire tile's temperature is kept so one thread can easily look up its neighbours temp, and not have to go to global mem.
    //Note that here we have extra space for the halo. it loads the next block's first thread's data and the prev block's last thread's data.
    //Idea is to step once into time make changes to temperature and then write back to global memory, and then go for the next time step.

    int halo_offset = TOTAL_HALOS/2;                                       //halo_offset tells where the data starts in shared memory.

    //---------------------------LOAD INITIAL U DATA FROM GLOBAL MEMORY TO SHARED MEMORY---------------------------//
    sharedtemp[halo_offset + threadIdx.x] = U[0*TOTAL_PARTS + xid];             //load the initial U values for this iteration of the kernel
    if (threadIdx.x<halo_offset)                                                //threads that need to load the halo values.
    {
        if (blockIdx.x!=0)                                                      //load prev halo values
            sharedtemp[threadIdx.x] = U[0*TOTAL_PARTS + xid-halo_offset];
        else
            sharedtemp[threadIdx.x] = BOUNDARY_TEMP;

        if (blockIdx.x!=TOTAL_BLOCKS-1)
            sharedtemp[threadIdx.x + halo_offset + TILESIZE] = U[0*TOTAL_PARTS + xid+TILESIZE];     //TODO:URGENT ERROR CHECK THIS [ITS CORRECT]
        else
            sharedtemp[threadIdx.x + halo_offset + TILESIZE] = BOUNDARY_TEMP;
    }

    __syncthreads();
    //---------------------------START TIME STEPPING LOCALLY IN THE CURRENT KERNEL, USING HALOS---------------------------//

    for(int time=1;time<=halo_offset;time++)
    {
        /*time=0 was the initial condition that we loaded in from global mem. Note time<=halo. equal to signmakes life easy, since we get HALO/2 new rod data.
        Note that U has rows upto HALO/2 + 1. the +1 is for the initial conditions to be loaded at row0. thereafter we have HALO/2 new steps to calculate. and TOTAL_TIME is
        perfectly divisble by HALO/2. So we chill.
        */
        float new_curr1 = 0;
        float new_curr2 = 0;
        //calculate the new halo values first.
        if (threadIdx.x < halo_offset)
        {
            float curr1 = sharedtemp[threadIdx.x];                      //curr1, next1, prev1 are for 1st halo part
            float next1 = sharedtemp[threadIdx.x+1];
            float prev1 = 0;
            if (threadIdx.x>0)                                          //prev for 1st halo is BOUNDARY_TEMP
                prev1 = sharedtemp[threadIdx.x-1];
            else
                prev1 = BOUNDARY_TEMP;

            new_curr1 = curr1 + R*(next1+prev1 - 2*curr1);

            float curr2 = sharedtemp[halo_offset+TILESIZE+threadIdx.x];         //curr2, next2, prev2 are for the 2nd halo part after the main TILE.
            float next2 = 0;
            if (threadIdx.x<halo_offset-1)                                      //next for last halo is BOUNDARY TEMP
                next2 = sharedtemp[halo_offset+TILESIZE+threadIdx.x+1];
            else
                next2 = BOUNDARY_TEMP;

            float prev2 = sharedtemp[halo_offset+TILESIZE+threadIdx.x-1];

            new_curr2 = curr2 + R*(prev2 + next2 - 2*curr2);

        }

        float curr = sharedtemp[halo_offset+threadIdx.x];
        float next = sharedtemp[halo_offset+threadIdx.x+1];
        float prev = sharedtemp[halo_offset+threadIdx.x-1];

        float new_curr = curr + R*(prev + next - 2*curr);

        __syncthreads();                //SYNC BEFORE WRITING TO SHARED MEMORY. [BUG FIXED]


        sharedtemp[halo_offset+threadIdx.x] = new_curr;
        if (threadIdx.x<halo_offset)                                            //Write new halo values to the shared memory, for next iter.
        {
            sharedtemp[threadIdx.x] = new_curr1;
            sharedtemp[halo_offset+TILESIZE+threadIdx.x] = new_curr2;
        }

        __syncthreads();                 // REINTRODUCED THIS SYNC, SINCE WE LOOP BACK TO READING, SO WRITING MUST BE FINISHED.
    }

    U[1*TOTAL_PARTS + xid] = sharedtemp[halo_offset+threadIdx.x];        //Write to global memory

}

int main()
{
    float * h_arr, *initial_rod_temp;
    float * h_arr2;
    int memory1d = TOTAL_PARTS*sizeof(float);         // the amount of memory needed to store the 1d array temp info of the rod.
    //h_arr = (float*) malloc(TOTAL_TIME*memory1d);
    h_arr2 = (float*)malloc(memory1d*TOTAL_TIME_DATAPOINT_ROWS);
    initial_rod_temp = (float*) malloc(memory1d);

    float * d_U;
    cudaMalloc(&d_U, 2*memory1d);               //dimension goes down only to 2.

    dim3 blocksize(TILESIZE);
    dim3 gridsize(TOTAL_BLOCKS);
    initialize_temp<<<gridsize, blocksize>>>(d_U);
    cudaMemcpy(initial_rod_temp, d_U, memory1d, cudaMemcpyDeviceToHost);

    for(int time_stride=0; time_stride<TOTAL_TIME_DATAPOINT_ROWS; time_stride++)
    {
        Heat_PDE_1D<<<gridsize, blocksize>>>(d_U);                  //d_U[-1] needs to be stored into d_U[0] for the next iteration. In the mean time copy the data into host.
        cudaMemcpy(&h_arr2[time_stride*(TOTAL_PARTS)], &d_U[1*TOTAL_PARTS], memory1d, cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_U[0], &d_U[1*TOTAL_PARTS], memory1d, cudaMemcpyDeviceToDevice);
    }

    FILE *f;
    f = fopen("rod_temp_data_skip80.csv", "w");

    for (int row=0; row<TOTAL_TIME_DATAPOINT_ROWS; row++)
    {
        for(int posx=0; posx<TOTAL_BLOCKS*TILESIZE; posx+=80)
        {
            fprintf(f, "%.1f,", h_arr2[row*TOTAL_PARTS + posx]);
        }
        fprintf(f, "%d\n", BOUNDARY_TEMP );
    }


    fclose(f);
    free(h_arr2);
    free(initial_rod_temp);
    cudaFree(d_U);

    return 0;
}


