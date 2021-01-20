import torch
from absl import app
from absl import flags
from data import tigdog_final as tf_final
from data import frame_loader as tigdog_mf
from Models import UNet, xy_outputs
import pickle as pkl
import os
import os.path as osp
from torch.utils.data import DataLoader
import sklearn.linear_model
import numpy as np

flags.DEFINE_integer('img_size', 128, '')
flags.DEFINE_string('root_dir', '/home/xindeik/data/TigDog_new_wnrsfm_new/', 'tmp dir to extract dataset')
flags.DEFINE_string('category', 'tiger', 'tmp dir to extract dataset')
flags.DEFINE_string('tb_log_dir', 'logs/', 'tmp dir to extract dataset')
flags.DEFINE_integer('num_kps', 18, '')
flags.DEFINE_integer('vis_every', 50, '')
flags.DEFINE_integer('num_frames', 2, '')
flags.DEFINE_integer('batch_size', 2, '')
flags.DEFINE_string('tmp_dir_train', 'saved_frames_train/', 'tmp dir to extract train dataset')
flags.DEFINE_string('tmp_dir_test', 'saved_frames_test/', 'tmp dir to extract test dataset')
flags.DEFINE_string('model_state_dir','state_dict_model0.pt', 'tmp dir to save model state')
opts = flags.FLAGS

def convert_landmarks(result_outputs):
    result_x, result_y = xy_outputs(result_outputs, scaling=False, scale=result_outputs.shape[-1])
    result = torch.stack((result_x, result_y), dim=0).view(opts.batch_size, -1).t().contiguous().view(opts.batch_size, -1)
    return result

def main(_):

    torch.manual_seed(0)
    if opts.category in ['horse', 'tiger']:
        dataset_train = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=False, normalize=False,
                                               max_length=None, remove_neck_kp=False, split='train',
                                               img_size=opts.img_size, mirror=False, scale=False, crop=False)

        dataset_test = tf_final.TigDogDataset_Final(opts.root_dir, opts.category, transforms=False, normalize=False,
                                               max_length=None, remove_neck_kp=False, split='test',
                                               img_size=opts.img_size, mirror=False, scale=False, crop=False)

        collate_fn = tf_final.TigDog_collate

    directory_train = opts.tmp_dir_train + '/' + opts.category + '/'
    if not osp.exists(directory_train):
        os.makedirs(directory_train)

    directory_test = opts.tmp_dir_test + '/' + opts.category + '/'
    if not osp.exists(directory_test):
        os.makedirs(directory_test)

    save_counter_train = 0
    sample_to_vid_train = {}
    samples_per_vid_train = {}
    print('Number of training videos for ', opts.category, '-', len(dataset_train))
    i_sample = 0
    for i_sample, sample in enumerate(dataset_train):
        num_frames = sample['video'].shape[0]
        for i in range(num_frames):
            new_sample = {}
            for k in sample.keys():
                if k in ['video', 'sfm_poses', 'landmarks', 'segmentations', 'bboxes']:
                    new_sample[k] = sample[k][i]

            pkl.dump(new_sample, open(directory_train + str(save_counter_train) + '.pkl', 'wb'))
            sample_to_vid_train[save_counter_train] = i_sample
            if i_sample in samples_per_vid_train:
                samples_per_vid_train[i_sample].append(save_counter_train)
            else:
                samples_per_vid_train[i_sample] = [save_counter_train]
            save_counter_train += 1
           # if i >= 5:  # 35:  # TODO:fix this
               # break
        #if i_sample >= 3:  # TODO:fix this
           # break

    training_samples = save_counter_train
    print('Training samples (frames):', training_samples)


    save_counter_test = 0
    sample_to_vid_test = {}
    samples_per_vid_test = {}
    print('Number of testing videos for ', opts.category, '-', len(dataset_test))
    i_sample = 0
    for i_sample, sample in enumerate(dataset_test):
        num_frames = sample['video'].shape[0]
        for i in range(num_frames):
            new_sample = {}
            for k in sample.keys():
                if k in ['video', 'sfm_poses', 'landmarks', 'segmentations', 'bboxes']:
                    new_sample[k] = sample[k][i]

            pkl.dump(new_sample, open(directory_test + str(save_counter_test) + '.pkl', 'wb'))
            sample_to_vid_test[save_counter_test] = i_sample
            if i_sample in samples_per_vid_test:
                samples_per_vid_test[i_sample].append(save_counter_test)
            else:
                samples_per_vid_test[i_sample] = [save_counter_test]
            save_counter_test += 1
           # if i >= 5:  # 35:  # TODO:fix this
               # break
        #if i_sample >= 3:  # TODO:fix this
           # break

    testing_samples = save_counter_test
    print('Testing samples (frames):', testing_samples)

    dataset_train = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir_train, opts.category, num_frames=opts.num_frames,
                                                 sample_to_vid=sample_to_vid_train,
                                                 samples_per_vid=samples_per_vid_train,
                                                 normalize=True, transforms=True,
                                                 remove_neck_kp=True, split='train', img_size=opts.img_size,
                                                 mirror=True, scale=True, crop=True, v2_crop=True, tight_bboxes=True)

    collate_fn = tigdog_mf.TigDog_collate

    dataloader_train = DataLoader(dataset_train, opts.batch_size, drop_last=True, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)


    print('Dataloader Train:', len(dataloader_train))


    dataset_test = tigdog_mf.TigDogDataset_MultiFrame(opts.tmp_dir_test, opts.category, num_frames=opts.num_frames,
                                                 sample_to_vid=sample_to_vid_test,
                                                 samples_per_vid=samples_per_vid_test,
                                                 normalize=True, transforms=True,
                                                 remove_neck_kp=True, split='test', img_size=opts.img_size,
                                                 mirror=True, scale=True, crop=True, v2_crop=True, tight_bboxes=True)


    dataloader_test = DataLoader(dataset_test, opts.batch_size, drop_last=True, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

    print('Dataloader Test:', len(dataloader_test))

    with torch.no_grad():

        keypoint_model = UNet(opts.num_kps).cuda()
        checkpoint = torch.load(opts.model_state_dir)
        keypoint_model.load_state_dict(checkpoint['keypoint_state_dict'])
        n_iter_test = 0
        n_iter_train = 0
        train_num = len(dataloader_train)*2
        test_num = len(dataloader_test)*2
        X_train = torch.zeros(train_num, opts.num_kps*2).cpu()
        y_train = torch.zeros(train_num, opts.num_kps*2).cpu()
        X_test = torch.zeros(test_num, opts.num_kps*2).cpu()
        y_test = torch.zeros(test_num, opts.num_kps*2).cpu()
        regr = sklearn.linear_model.Ridge(alpha=0.0, fit_intercept=False)
        for sample in dataloader_train:
            input_img_tensor = sample['img'].type(torch.FloatTensor).clone().cuda()
            frame1 = input_img_tensor[:, 0]
            frame2 = input_img_tensor[:, 1]
            target = frame2
            source = frame1
            target_outputs = keypoint_model(target).cpu()
            keypoints_pred = convert_landmarks(target_outputs).cpu()
            keypoints_gt = sample['kp'].type(torch.FloatTensor)[:,1,:,:2]
            keypoints_gt = ((keypoints_gt + 1) / 2.0) * opts.img_size
            keypoints_gt = keypoints_gt.reshape((opts.batch_size,-1))
            print(keypoints_pred, keypoints_gt)
            X_train[n_iter_train:n_iter_train+2,:] = keypoints_pred
            y_train[n_iter_train:n_iter_train+2,:] = keypoints_gt
            n_iter_train += 2

        _ = regr.fit(X_train.detach().numpy(), y_train.detach().numpy())

        for sample in dataloader_test:
            input_img_tensor = sample['img'].type(torch.FloatTensor).clone().cuda()
            frame1 = input_img_tensor[:, 0]
            frame2 = input_img_tensor[:, 1]
            target = frame2
            source = frame1
            target_outputs = keypoint_model(target).cpu()
            keypoints_pred = convert_landmarks(target_outputs).cpu()
            keypoints_gt = sample['kp'].type(torch.FloatTensor)[:,1,:,:2]
            keypoints_gt = ((keypoints_gt + 1) / 2.0) * opts.img_size
            keypoints_gt = keypoints_gt.reshape((opts.batch_size,-1))
            X_test[n_iter_test:n_iter_test+2,:] = keypoints_pred
            y_test[n_iter_test:n_iter_test+2,:] = keypoints_gt
            n_iter_test += 2

        y_predict = regr.predict(X_test.detach().numpy())
        keypoints_gt = y_test.reshape(test_num, opts.num_kps, 2).cpu().detach().numpy()
        keypoints_regressed = y_predict.reshape(keypoints_gt.shape)

        chin = keypoints_gt[:, 3, :]
        tailStart = keypoints_gt[:, 8, :]

        chin_tail_distances = np.sqrt(np.sum((chin - tailStart)**2, axis=-1))
        distances = np.sqrt(np.sum((keypoints_gt - keypoints_regressed)**2, axis=-1))
        print(chin_tail_distances)
        chin_tail_distances += 1e-8
        mean_error = np.mean(distances / chin_tail_distances[:, None])
        print(mean_error)

if __name__ == '__main__':
        app.run(main)
