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
        _scores = tf.squeeze(_scores, axis = 1)
        patch_scores = _scores.numpy()

        if isinstance(features, np.ndarray):
            features = tf.convert_to_tensor(features)
        features = tf.identity(features)
        features = tf.transpose(features, perm=([0,3,1,2]))
        if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2**31:
            subbatch_size = int((2**31-1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
            interpolated_features = []
            for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
                subfeatures = features[i_subbatch*subbatch_size:(i_subbatch+1)*subbatch_size]
                subfeatures = subfeatures.unsuqeeze(0) if len(subfeatures.shape) == 3 else subfeatures
                subfeatures = tf.image.resize(subfeatures, size = self.target_size, method=tf.image.ResizeMethod.BILINEAR)
                interpolated_features.append(subfeatures)
            features = tf.concat(interpolated_features, axis = 0)
        else:
            features = tf.image.resize(features, size = self.target_size, method = tf.image.ResizeMethod.BILINEAR)
        features = features.numpy()

        return [
        ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores
        ], [feature for feature in features]
    

class NetworkFeatureAggregator(tf.keras.Model):
    def __init__(self, backbone, layers_to_extract_from, train_backbone = False):
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.train_backbone = train_backbone
        self.outputs = {}

        self.hook_handles = []
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if '.' in extract_layer:
                extract_block, extract_idx = extract_layer.split('.')
                network_layer = backbone.get_layer(extract_block)
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer.get_layer(extract_idx)
                else:
                    network_layer = network_layer.get_layer(extract_idx)
            else:
                network_layer = backbone.get_layer(extract_layer)
            if isinstance(network_layer, tf.keras.Sequential):
                self.hook_handles.append(
                    network_layer.layers[-1].register_forward_hook(forward_hook)
                )

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_excepttion_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )
    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None