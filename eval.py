import numpy as np
import sklearn.linear_model
from data import tigdog_final as tf_final
from absl import flags
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import utils
import matplotlib.backends.backend_agg as plt_backend_agg

alpha_str = '0.1'
keypoint_dir = 'keypoint_testing_kp30_reconstruct_only.npz'
dataloader_dir = 'dataloader_testing'

test_num = 883 * 2
kps = 18
gauss_kps = 30

writer = SummaryWriter(log_dir='Groundtruth_Predicted_Testing_kp30_reconstruct_only')
writer2 = SummaryWriter(log_dir='Accuracy_Graph_Testing_kp40_reconstruct_affine_0.1')
npzfile = np.load(keypoint_dir)
regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)

X_train = npzfile['X_train']
y_train = npzfile['y_train']
X_test = npzfile['X_test']
y_test = npzfile['y_test']
test_image = npzfile['test_image']

print(X_train.shape)
_ = regr.fit(X_train, y_train)

y_predict = regr.predict(X_test)
keypoints_gt = y_test.reshape(test_num,kps, 3)
keypoints_gauss = X_test.reshape(test_num,gauss_kps,3)
keypoints_regressed = y_predict.reshape(keypoints_gt.shape)

for i in range(test_num):
    img = torch.tensor(test_image[i])
    gt = torch.tensor(keypoints_gt[i])
    gauss = torch.tensor(keypoints_gauss[i])
    regressed = torch.tensor(keypoints_regressed[i])
    gt_vis = torch.stack([gt[:,0],gt[:,1],gt[:,2]], dim=-1)

    visibility_baseline = 0.5

    for j in range(kps):
        visibility = regressed[j,2]
        if(visibility<visibility_baseline):
            #if visibility is lesser than baseline, set coordinate to center like orignal data
            regressed[j,0] = 0
            regressed[j,1] = 0
            regressed[j,2] = 0
            keypoints_regressed[i,j,0] = 0
            keypoints_regressed[i,j,1] = 0
            keypoints_regressed[i,j,2] = 0
        else:
            regressed[j,2] = 1
            keypoints_regressed[i,j,2] = 1

    gauss_vis = torch.stack([gauss[:,0],gauss[:,1],gauss[:,2]], dim=-1)
    regressed_vis = torch.stack([regressed[:,0],regressed[:,1],regressed[:,2]], dim=-1)
    kp_img_gt = utils.kp2im(gt_vis, img, radius=2)/255
    kp_img_gauss = utils.kp2im(gauss_vis, img, radius=2)/255
    kp_img_predict = utils.kp2im(regressed_vis, img, radius=2)/255
    kp_img_gt = torch.tensor(kp_img_gt).permute(2,0,1)[None]
    kp_img_gauss = torch.tensor(kp_img_gauss).permute(2,0,1)[None]
    kp_img_predict = torch.tensor(kp_img_predict).permute(2,0,1)[None]
    num_kps_gt = torch.count_nonzero(gt[:,2])
    num_kps_reg = torch.count_nonzero(regressed[:,2])
    accuracy = torch.eq(gt[:,2],regressed[:,2]).float().mean()
    regressed_01 = (regressed[:,:2].clone() + 1)/2
    gt_01 = (gt[:,:2].clone() + 1)/2
    mse = torch.sum((regressed_01 - gt_01)**2)
    grid = torch.cat([kp_img_gt, kp_img_gauss, kp_img_predict], dim=3)[0]
    writer.add_image('Image ' + str(i) + "ERR " + str(mse) + "GT " + str(num_kps_gt) + "REG " + str(num_kps_reg) +"ACC " + str(accuracy), grid, i)

writer.close()

#rescale keypoints from (-1,1) to (0,1)
keypoints_gt[:,:,:2] = (keypoints_gt[:,:,:2] + 1)/2
keypoints_gauss[:,:,:2] = (keypoints_gauss[:,:,:2] + 1)/2
keypoints_regressed[:,:,:2] = (keypoints_regressed[:,:,:2] + 1)/2

accuracy_kps_vis = np.equal(keypoints_gt[:,:,2],keypoints_regressed[:,:,2]).astype(float).mean()
accuracy_kps_wo_vis = np.equal(keypoints_gt[:,:,2], np.ones(keypoints_gt[:,:,2].shape)).astype(float).mean()

print("With visibility ", accuracy_kps_vis)
print("Without visibility ", accuracy_kps_wo_vis)

fl = keypoints_gt[:, 8, :]
bl = keypoints_gt[:, 10, :]
fr = keypoints_gt[:, 9, :]
br = keypoints_gt[:, 11, :]
l_distances = np.sqrt(np.sum((fl - bl)**2, axis=-1))
r_distances = np.sqrt(np.sum((fr - br)**2, axis=-1))
normalise_distances = (l_distances + r_distances)/2
distances = np.sqrt(np.sum((keypoints_gt[:,:,:2] - keypoints_regressed[:,:,:2])**2, axis=-1))

accurate_range = [i*0.05 for i in range(11)]
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
    ax.title.set_text(keypoints_name[index])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Within Pixel")
    canvas = plt_backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w,h = fig.canvas.get_width_height()
    img = data.reshape([h,w,4])[:,:,:3]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = torch.tensor(img)
    writer2.add_image(keypoints_name[index], img, index)


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
