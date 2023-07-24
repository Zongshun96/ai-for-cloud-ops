set -e


# backup data
for d in ./data*.zip ;
do
    echo $d "gdrive files upload '${d%/}'"
    gdrive files upload "${d%/}" --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
done

# gdrive files upload './m4_nop_reconstruction_client_5_equal_work_dataset_base_500_[0, 7]_cifar10_equDiff.zip' --parent 1cSgYLRJsrZlviG_JaelrzjxkOr6YQIpA
