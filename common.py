from lib import *

class _BaseMerger:
    def __init__(self):
        pass
    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis = 1)
    
class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(axis = -1)


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)

class Preprocessing(tf.keras.Model):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing,self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = list()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)
    
    def call(self, features):
        _features = list()
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return tf.stack(_features, axis = 1)

class MeanMapper(tf.keras.Model):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim
    
    def call(self, features):
        features = tf.reshape(features, (len(features), 1, -1))
        return tf.squeeze(layers.GlobalAveragePooling1D()(features), axis = 1)

class Aggregator(tf.keras.Model):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim
    
    def call(self, features):
        """Returns reshaped and average pooled features"""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = tf.reshape(features, (len(features), 1, -1))
        features = layers.GolbalAveragePooling1D()(features)
        return tf.reshape(features, (len(features), -1))

class RescaleSegmentor:
    def __init__(self, device, target_size = 224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4
    def convert_to_segmentaion(self, patch_scores, features):
        if isinstance(patch_scores, np.ndarray):
            path_scores = tf.convert_to_tensor(patch_scores)
        _scores = tf.identity(patch_scores)
        _scores = tf.expand_dims(_scores, axis = 1)
        _scores = tf.image.resize(_scores, 
                                  size = self.target_size, 
                                  method = tf.image.ResizeMethod.BILINEAR)



