import motmetrics as mm
import pandas as pd

gt = mm.io.loadtxt("gt.txt", fmt="mot15-2D", min_confidence=1)
ts = mm.io.loadtxt("tracker.txt", fmt="mot15-2D")

acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

mh = mm.metrics.create()

summary = mh.compute(acc, metrics=[
    'mota', 'idf1', 'num_false_positives',
    'num_misses', 'num_switches'
], name='YOLOv8_DeepSORT')

print(summary)