from __future__ import print_function

import numpy as np
import six
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def batch_generator(X, y, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


def batch_seq_generator(X, y, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    i = 0
    while True:
        yield X[i:i + batch_size], y[i:i + batch_size]
        if i + batch_size >= size:
            break
        else:
            i += batch_size


def san_batch_generator(X, y, attention_labels, batch_size):
    """Primitive batch generator 
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()
    X_copy = X_copy[indices]

    y_copy = y.copy()
    y_copy = y_copy[indices]

    if attention_labels is not None:
        attention_labels_copy = attention_labels.copy()
        attention_labels_copy = attention_labels_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if attention_labels is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            if attention_labels is not None:
                attention_labels_copy = attention_labels_copy[indices]
            continue


def great_batch_generator(X, y, attention_labels, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()
    X_copy = X_copy[indices]

    if y is not None:
        y_copy = y.copy()
        y_copy = y_copy[indices]

    if attention_labels is not None:
        attention_labels_copy = attention_labels.copy()
        attention_labels_copy = attention_labels_copy[indices]

    if masks is not None:
        masks_copy = masks.copy()
        masks_copy = masks_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if y is not None and attention_labels is not None and masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size], masks_copy[i:i+ batch_size]
            elif y is None and attention_labels is not None and masks is not None:
                yield X_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size], masks_copy[i:i+ batch_size]
            elif y is None and attention_labels is not None and masks is None:
                yield X_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            elif y is not None and attention_labels is not None and masks is None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], attention_labels_copy[i:i + batch_size]
            elif y is not None and attention_labels is None and masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], masks_copy[i:i+ batch_size]
            elif y is not None and attention_labels is None and masks is None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            if y is not None:
                y_copy = y_copy[indices]
            if attention_labels is not None:
                attention_labels_copy = attention_labels_copy[indices]
            if masks is not None:
                masks_copy = masks_copy[indices]
            continue


def mask_batch_generator(X, y, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    X_copy = X.copy()
    X_copy = X_copy[indices]

    y_copy = y.copy()
    y_copy = y_copy[indices]

    if masks is not None:
        masks_copy = masks.copy()
        masks_copy = masks_copy[indices]

    i = 0
    while True:
        if i + batch_size <= size:
            if masks is not None:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size], masks_copy[i:i + batch_size]
            else:
                yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            if masks is not None:
                masks_copy = masks_copy[indices]
            continue


def mask_batch_seq_generator(X, y, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]

    i = 0
    while True:
        if masks is not None:
            yield X[i:i + batch_size], y[i:i + batch_size], masks[i:i + batch_size]
        else:
            yield X[i:i + batch_size], y[i:i + batch_size]
        if i + batch_size >= size:
            break
        else:
            i += batch_size


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def eval_metrics(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    values, indices = torch.max(probabilities, 1)
    y_pred = indices
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, values)
    except ValueError:
        auc = np.array(0)
    conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    pos_f1_score = f1_score(y, y_pred, average='binary', zero_division=1)
    mar_f1_score = f1_score(y, y_pred, average='macro', zero_division=1)
    mic_f1_score = f1_score(y, y_pred, average='micro', zero_division=1)
    wei_f1_score = f1_score(y, y_pred, average='weighted', zero_division=1)
    performance_dict = {'acc':acc, 'auc':auc,
                        'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp,
                        'pos_f1':pos_f1_score, 'mac_f1':mar_f1_score,
                        'mic_f1':mic_f1_score, 'wei_f1':wei_f1_score}
    return performance_dict


def eval_predicted_metrics(y_pred, values, y):
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, values)
    except ValueError:
        auc = np.array(0)
    conf_mat = confusion_matrix(y, y_pred, labels=[0, 1])
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    pos_f1_score = f1_score(y, y_pred, average='binary', zero_division=1)
    mar_f1_score = f1_score(y, y_pred, average='macro', zero_division=1)
    mic_f1_score = f1_score(y, y_pred, average='micro', zero_division=1)
    wei_f1_score = f1_score(y, y_pred, average='weighted', zero_division=1)
    performance_dict = {'acc': acc, 'auc': auc,
                        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                        'pos_f1': pos_f1_score, 'mac_f1': mar_f1_score,
                        'mic_f1': mic_f1_score, 'wei_f1': wei_f1_score}
    return performance_dict


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8
    """
    m = nn.Softmax(dim=1)
    probabilities = m(preds)
    values, indices = torch.max(probabilities, 1)
    y_pred = indices
    acc = accuracy_score(y, y_pred)
    return acc


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array(['a', 'b', 'c', 'd']), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = next(gen)
        print(xx, yy)
