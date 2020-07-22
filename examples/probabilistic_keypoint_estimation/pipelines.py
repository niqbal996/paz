from paz.backend.keypoints import denormalize_keypoints
from paz.abstract import SequentialProcessor
from paz.abstract import Processor
from paz import processors as pr


from processors import PartitionKeypoints
from processors import PredictMeanDistribution
from processors import draw_circles


class AugmentKeypoints(SequentialProcessor):
    def __init__(self, phase, rotation_range=30, delta_scales=[0.2, 0.2],
                 with_partition=False, num_keypoints=15):
        super(AugmentKeypoints, self).__init__()
        self.add(pr.UnpackDictionary(['image', 'keypoints']))
        if phase == 'train':
            self.add(pr.ControlMap(pr.RandomBrightness()))
            self.add(pr.ControlMap(pr.RandomContrast()))
            self.add(pr.RandomKeypointRotation(rotation_range))
            self.add(pr.RandomKeypointTranslation(delta_scales))
        self.add(pr.ControlMap(pr.NormalizeImage(), [0], [0]))
        self.add(pr.ControlMap(pr.ExpandDims(-1), [0], [0]))
        self.add(pr.ControlMap(pr.NormalizeKeypoints((96, 96)), [1], [1]))
        labels_info = {1: {'keypoints': [num_keypoints, 2]}}
        if with_partition:
            outro_indices = list(range(1, 16))
            self.add(pr.ControlMap(PartitionKeypoints(), [1], outro_indices))
            labels_info = {}
            for arg in range(num_keypoints):
                labels_info[arg + 1] = {'keypoint_%s' % arg: [2]}
        self.add(pr.SequenceWrapper(
            {0: {'image': [96, 96, 1]}}, labels_info))


class ProbabilisticKeypointPrediction(Processor):
    def __init__(self, detector, keypoint_estimator, radius=3):
        super(ProbabilisticKeypointPrediction, self).__init__()
        # face detector
        RGB2GRAY = pr.ConvertColorSpace(pr.RGB2GRAY)
        self.detect = pr.Predict(detector, RGB2GRAY, pr.ToBoxes2D(['face']))

        # creating pre-processing pipeline for keypoint estimator
        preprocess = SequentialProcessor()
        preprocess.add(pr.ResizeImage(keypoint_estimator.input_shape[1:3]))
        preprocess.add(pr.ConvertColorSpace(pr.RGB2GRAY))
        preprocess.add(pr.NormalizeImage())
        preprocess.add(pr.ExpandDims([0, 3]))

        # creating post-processing pipeline for keypoint esimtator
        # postprocess = SequentialProcessor()
        # postprocess.add(ToNumpyArray())
        # postprocess.add(pr.Squeeze(1))

        # keypoint estimator predictions
        self.estimate_keypoints = PredictMeanDistribution(
            keypoint_estimator, preprocess)

        # self.estimate_keypoints = pr.Predict(
        # keypoint_estimator, preprocess, postprocess)

        # used for drawing up keypoints in original image
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.denormalize_keypoints = pr.DenormalizeKeypoints()
        self.crop_boxes2D = pr.CropBoxes2D()
        self.num_keypoints = len(keypoint_estimator.output_shape)
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, radius, False)
        self.draw_boxes2D = pr.DrawBoxes2D(['face'], colors=[[0, 255, 0]])
        self.wrap = pr.WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.detect(image)
        cropped_images = self.crop_boxes2D(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            keypoints = self.estimate_keypoints(cropped_image)
            keypoints = self.denormalize_keypoints(keypoints, cropped_image)
            keypoints = self.change_coordinates(keypoints, box2D)
            image = self.draw(image, keypoints)
        image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


if __name__ == '__main__':
    from paz.abstract import ProcessingSequence
    from paz.backend.image import show_image

    from facial_keypoints import FacialKeypoints

    data_manager = FacialKeypoints('dataset/', 'train')
    dataset = data_manager.load_data()
    augment_keypoints = AugmentKeypoints('train', with_partition=False)
    for arg in range(1, 100):
        sample = dataset[arg]
        predictions = augment_keypoints(sample)
        original_image = predictions['inputs']['image'][:, :, 0]
        original_image = original_image * 255.0
        kp = predictions['labels']['keypoints']
        kp = denormalize_keypoints(kp, 96, 96)
        original_image = draw_circles(
            original_image, kp.astype('int'))
        show_image(original_image.astype('uint8'))
    sequence = ProcessingSequence(augment_keypoints, 32, dataset, True)
    batch = sequence.__getitem__(0)
