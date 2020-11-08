
data_name="006FL.MTS" # (keypoints to use)
subject="006FL.MTS"
out_video="/home/filipkr/Documents/xjob/results-w-meta/results/outputFL2.mp4"
in_video="/home/filipkr/Documents/xjob/results-w-meta/results/vis_006FL25HRNetTopDownCocoDataset.mp4"
out_data="/home/filipkr/Documents/xjob/results-FL/results/006FL_3D2.npy"


python run.py -d custom -k $data_name -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject $subject --viz-action custom --viz-camera 0 --viz-video $in_video --viz-output $out_video --viz-size 6 --viz-export $out_data -no-tta -no-da

