export CUDA_VISIBLE_DEVICES=2
# sizes='32 64 128 256 512 1024 2048'
# sizes='256'
sizes='1024'

#coordinates='4'
#coordinates='16'
coordinates='32'

#centers='64'
#centers='16'
centers='64'

loop_threashold='10'
# loop_threashold='100''

block_size='32 48 64 128 238 512 1024'

progs=(
#       kmeans_seq
#       kmeans_cuda_naive
#       kmeans_cuda_transpose
#       kmeans_cuda_shared
        kmeans_cuda_all_gpu
)


for size in $sizes; do
        for coord in $coordinates; do
                for center in $centers; do
                        filename=Execution_logs/Sz-${size}_Coo-${coord}_Cl-${center}.csv
                        echo "Implementation,blockSize,av_loop_t,min_loop_t,max_loop_t" >> $filename
                        for prog in "${progs[@]}"; do
                                if [[ $prog == 'kmeans_seq' ]]; then
                                        ./${prog} -s $size -n $coord -c $center -l $loop_threashold
                                fi
                                for bs in $block_size; do
                                        if [[ $prog == 'kmeans_cuda_naive' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_transpose' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_shared' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_all_gpu' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_all_gpu_delta_reduction' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_all_gpu_reduction' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        elif [[ $prog == 'kmeans_cuda_gitman' ]]; then
                                                ./${prog} -s $size -n $coord -c $center -l $loop_threashold -b $bs
                                        fi
                                done
                        done
                done
        done
done
