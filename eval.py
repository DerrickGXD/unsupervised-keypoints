import numpy as np
import sklearn.linear_model
from data import tigdog_final as tf_final
from absl import flags
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import utils
import matplotlib.backends.backend_agg as plt_backend_agg

alpha_str = '0.01'
keypoint_dir = 'keypoint_affine_' + alpha_str + '.npz'
dataloader_dir = 'dataloader_test_' + alpha_str

test_num = 883 * 2
kps = 18

writer = SummaryWriter(log_dir='Groundtruth_Predicted_' + alpha_str)
writer = SummaryWriter(log_dir='Accuracy_Graph_Affine' + alpha_str)
npzfile = np.load(keypoint_dir)
regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)

X_train = npzfile['X_train']
y_train = npzfile['y_train']
X_test = npzfile['X_test']
y_test = npzfile['y_test']
test_image = npzfile['test_image']

_ = regr.fit(X_train, y_train)

y_predict = regr.predict(X_test)
keypoints_gt = y_test.reshape(test_num,kps, 2)
keypoints_regressed = y_predict.reshape(keypoints_gt.shape)

for i in range(test_num):
    img = torch.tensor(test_image[i])
    gt = torch.tensor(keypoints_gt[i])
    regressed = torch.tensor(keypoints_regressed[i])
    gt = (gt*2/128) -1
    regressed = (regressed*2/128)-1
    gt_vis = torch.stack([gt[:,0],gt[:,1],torch.ones_like(gt[:,1])], dim=-1)
    regressed_vis = torch.stack([regressed[:,0],regressed[:,1],torch.ones_like(regressed[:,1])], dim=-1)
    kp_img_gt = utils.kp2im(gt_vis, img, radius=2)/255
    kp_img_predict = utils.kp2im(regressed_vis, img, radius=2)/255
    kp_img_gt = torch.tensor(kp_img_gt).permute(2,0,1)[None]
    kp_img_predict = torch.tensor(kp_img_predict).permute(2,0,1)[None]
    grid = torch.cat([kp_img_gt, kp_img_predict], dim=3)[0]
    writer.add_image('Image ' + str(i) + ' std : ' + alpha_str, grid, i)

writer.close()
fl = keypoints_gt[:, 8, :]
bl = keypoints_gt[:, 10, :]
fr = keypoints_gt[:, 9, :]
br = keypoints_gt[:, 11, :]

l_distances = np.sqrt(np.sum((fl - bl)**2, axis=-1))
r_distances = np.sqrt(np.sum((fr - br)**2, axis=-1))
normalise_distances = (l_distances + r_distances)/2
distances = np.sqrt(np.sum((keypoints_gt - keypoints_regressed)**2, axis=-1))
#distances = 1/(1+distances)

accurate_range = [i for i in range(36)]
keypoints_name = {0:'leftEye', 1:'rightEye', 2:'chin', 3:'frontLeftFoot',
4:'frontRightFoot', 5:'backLeftFoot', 6:'backRightFoot', 7:'tailStart',
8:'frontLeftKnee', 9:'frontRightKnee', 10:'backLeftKnee', 11:'backRightKnee',
12:'leftShoulder', 13:'rightShoulder', 14:'frontLeftAnkle', 15:'frontRightAnkle',
16:'backLeftAnkle', 17:'backRightAnkle'}

for index in range(kps):
    accuracy_score = []
    for i in accurate_range:
        feature_distance = distances[:,index]
        feature_distance = np.where(feature_distance > i, 0, 1)
        accuracy = np.mean(feature_distance) * 100
        accuracy_score.append(accuracy)
    fig, ax  = plt.subplots()
    ax.plot(accurate_range, accuracy_score)
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w,h = fig.canvas.get_width_height()
    img = data.reshape([h,w,4])[:,:,:3]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = torch.tensor(img)
    writer.add_image(keypoints_name[index], img, index)


print("Error (Left Eye) : ", np.mean(distances[:,0]))
print("Error (Right Eye) : ", np.mean(distances[:,1]))
print("Error (Chin) : ", np.mean(distances[:,2]))
print("Error (Front Left Foot) : ", np.mean(distances[:,3]))
print("Error (Front Right Foot) : ", np.mean(distances[:,4]))
print("Error (Back Left Foot) : ", np.mean(distances[:,5]))
print("Error (Back Right Foot) : ", np.mean(distances[:,6]))
print("Error (Tail Start) : ", np.mean(distances[:,7]))
print("Error (Front Left Knee) : ", np.mean(distances[:,8]))
print("Error (Front Right Knee) : ", np.mean(distances[:,9]))
print("Error (Back Left Knee) : ", np.mean(distances[:,10]))
print("Error (Back Right Knee) : ", np.mean(distances[:,11]))
print("Error (Left Shoulder) : ", np.mean(distances[:,12]))
print("Error (Right Shoulder) : ", np.mean(distances[:,13]))
print("Error (Front Left Ankle) : ", np.mean(distances[:,14]))
print("Error (Front Right Ankle) : ", np.mean(distances[:,15]))
print("Error (Back Left Ankle) : ", np.mean(distances[:,16]))
print("Error (Back Right Ankle) : ", np.mean(distances[:,17]))

mean_distances = np.mean(distances)
print("Error without Normalised : ", mean_distances)
mean_error = np.mean(distances / normalise_distances[:, None])
print("Error with Normalised : ", mean_error)
