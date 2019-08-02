## RESNET50 on mouse32 runtimes
```
$ identify /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/1.2/1.2_Z0130.tif
/hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/1.2/1.2_Z0130.tif TIFF 9793x8585 9793x8585+0+0 8-bit Grayscale Gray 33.39MB 0.000u 0:00.000
$ /usr/bin/time -v deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/1.2/1.2_Z0130.tif --segment-output 1.2_Z0130_segment.tif --outlines-output 1.2_Z0130_outlines.tif --centroids-output 1.2_Z0130_centroids.tif --image-output 1.2_Z0130_image.tif
	Command being timed: "deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/1.2/1.2_Z0130.tif --segment-output 1.2_Z0130_segment.tif --outlines-output 1.2_Z0130_outlines.tif --centroids-output 1.2_Z0130_centroids.tif --image-output 1.2_Z0130_image.tif"
	User time (seconds): 20611.62
	System time (seconds): 7646.74
	Percent of CPU this job got: 256%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:03:52
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 5057840
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 703038666
	Voluntary context switches: 1645429
	Involuntary context switches: 2836536
	Swaps: 0
	File system inputs: 0
	File system outputs: 238432
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

$ identify /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/444.3/444.3_Z0050.tif
/hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/444.3/444.3_Z0050.tif TIFF 9666x8318 9666x8318+0+0 8-bit Grayscale Gray 47.85MB 0.000u 0:00.000
$ /usr/bin/time -v deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/444.3/444.3_Z0050.tif --segment-output 444.3_Z0050_segment.tif --outlines-output 444.3_Z0050_outlines.tif --centroids-output 444.3_Z0050_centroids.tif --image-output 444.3_Z0050_image.tif
	Command being timed: "deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/444.3/444.3_Z0050.tif --segment-output 444.3_Z0050_segment.tif --outlines-output 444.3_Z0050_outlines.tif --centroids-output 444.3_Z0050_centroids.tif --image-output 444.3_Z0050_image.tif"
	User time (seconds): 20213.60
	System time (seconds): 7795.81
	Percent of CPU this job got: 255%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:02:54
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 5004176
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 717623175
	Voluntary context switches: 1614722
	Involuntary context switches: 2435315
	Swaps: 0
	File system inputs: 0
	File system outputs: 326040
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0

$ identify /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/441.3/441.3_Z0100.tif
/hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/441.3/441.3_Z0100.tif TIFF 9672x8295 9672x8295+0+0 8-bit Grayscale Gray 48.11MB 0.000u 0:00.010
$ /usr/bin/time -v deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/441.3/441.3_Z0100.tif --segment-output 441.3_Z0100_segment.tif --outlines-output 441.3_Z0100_outlines.tif --centroids-output 441.3_Z0100_centroids.tif --image-output 441.3_Z0100_image.tif
	Command being timed: "deep_segment.py --learner /hpf/largeprojects/MICe/nwang/tools/fastai/2019-07-01_RESNET50_IOU0.69.pkl --image /hpf/largeprojects/MICe/nwang/TissueVision/2019-02-29_Mallar/nuclei_8brains_stitched/441.3/441.3_Z0100.tif --segment-output 441.3_Z0100_segment.tif --outlines-output 441.3_Z0100_outlines.tif --centroids-output 441.3_Z0100_centroids.tif --image-output 441.3_Z0100_image.tif"
	User time (seconds): 20252.12
	System time (seconds): 7765.16
	Percent of CPU this job got: 254%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 3:03:11
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 4956928
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 0
	Minor (reclaiming a frame) page faults: 714040632
	Voluntary context switches: 1636244
	Involuntary context switches: 2501379
	Swaps: 0
	File system inputs: 0
	File system outputs: 331184
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```
## RESNET34 on Graham (ComputeCanada) runtimes
```
$ cat deep_segment_job.sh
#!/bin/bash
#SBATCH --account=def-jlerch
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
time -v deep_segment.py --verbose --segment-intensity 1 --temp-dir temp --learner ~/tools/2019-07-24_RESNET34_IOU0.69_stage2.pkl --image TV_recon_test_stitched/447.3/447.3_Z0007.tif --image-output TV_recon_test_deep_segmentation/447.3/447.3_Z0007_cropped.tiff --centroids-output TV_recon_test_deep_segmentation/447.3/447.3_Z0007_count.tiff
$ sbatch deep_segment_job.sh
    Command being timed: "deep_segment.py --verbose --segment-intensity 1 --temp-dir temp --learner /home/nzxwang/tools/2019-07-24_RESNET34_IOU0.69_stage2.pkl --image TV_recon_test_stitched/447.3/447.3_Z0007.tif --image-output TV_recon_test_deep_segmentation/447.3/447.3_Z0007_cropped.tiff --centroids-output TV_recon_test_deep_segmentation/447.3/447.3_Z0007_count.tiff"
    User time (seconds): 1807.57
    System time (seconds): 93.11
    Percent of CPU this job got: 185%
    Elapsed (wall clock) time (h:mm:ss or m:ss): 17:02.17
    Average shared text size (kbytes): 0
    Average unshared data size (kbytes): 0
    Average stack size (kbytes): 0
    Average total size (kbytes): 0
    Maximum resident set size (kbytes): 1690700
    Average resident set size (kbytes): 0
    Major (requiring I/O) page faults: 982
    Minor (reclaiming a frame) page faults: 7831709
    Voluntary context switches: 32982
    Involuntary context switches: 134996
    Swaps: 0
    File system inputs: 1485584
    File system outputs: 138296
    Socket messages sent: 0
    Socket messages received: 0
    Signals delivered: 0
    Page size (bytes): 4096
    Exit status: 0
```