

for d in *-changesets/ ; do
    echo "mv $d /home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/data/"
    mv $d /home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/data/
    # rm -fr $d
done

for d in *-tagsets/ ; do
    echo "cp $d* /home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_train_tag/"
    cp $d* /home/cc/Praxi-study/praxi/demos/ic2e_demo/demo_tagsets/mix_train_tag/
    mv $d /home/cc/Praxi-study/ai-for-cloud-ops/RTQA/iPython/Praxi/data/
    # rm -fr $d
done