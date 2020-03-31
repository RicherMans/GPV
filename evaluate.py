import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
from models import crnn
import sed_eval
import os

SAMPLE_RATE = 22050  # Default resample-rate using audioread
EPS = np.spacing(1)  # Log zero division offset
DEVICE = 'cpu'  # Default run on CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    wav, sr = sf.read(wavefilepath, dtype='float32')
    # Multiple channels.. just go back to mono
    if wav.ndim > 1:
        wav = wav.mean(-1)
    # Resample in case of != 22.05k
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), sr, **kwargs) +
        EPS).T


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file


def event_based_evaluation_df(reference,
                              estimated,
                              t_collar=0.200,
                              percentage_of_length=0.2):
    """
    Calculate EventBasedMetric given a reference and estimated dataframe
    :param reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    reference events
    :param estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
    estimated events to be compared with reference
    :return: sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score')

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference,
                                estimated,
                                time_resolution=0.01):  # 10ms
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution)

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname)

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file)

    return segment_based_metric


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


## ALl of those are unavailable
## .wavlist contains an absolute pth specifiying each individual filename
## label contains the groun truth tab separated files in DCASE18 format (filename   onset offset    event_label)

TASKS = {
    'aurora_clean': {
        'data': 'aurora4_clean.wavlist',
        'label': 'aurora_clean_labels.tsv',
    },
    'aurora_noisy': {
        'data': 'aurora4_noise.wavlist',
        'label': 'aurora_noisy_labels.tsv'
    },
    'dcase18': {
        'data': 'dcase18.wavlist',
        'label': 'dcase18.tsv',
    },
}

MODELS = {
    'gpvf': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'label_encoders/gpv_f.pth',
        'pretrained': 'pretrained/gpv_f.pth',
        'resolution': 0.02
    },
    'gpvb': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'label_encoders/gpv_b.pth',
        'pretrained': 'pretrained/gpv_b.pth',
        'resolution': 0.02
    },
    'vadc': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'label_encoders/vad_c.pth',
        'pretrained': 'pretrained/vad_c.pth',
        'resolution': 0.02
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task',
                        choices=list(TASKS.keys()),
                        default='aurora4',
                        nargs="?")
    parser.add_argument('-model', choices=list(MODELS.keys()), default='gpvf')
    parser.add_argument('-n_mels', default=64, type=int)
    parser.add_argument('-n_fft',
                        default=2048,
                        type=int,
                        help='window size for fft, default %(default)')
    parser.add_argument('-hop_length', default=0.02, type=float, help='帧移')
    parser.add_argument('-win_length',
                        default=0.04,
                        type=float,
                        help='Window Length')
    parser.add_argument('-t', '--time_resolution', default=0.01, type=float)
    parser.add_argument('-o',
                        '--output_path',
                        default='results',
                        help='Base directory to dump results',
                        type=Path)
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.75, 0.2),
                        type=float,
                        nargs="+")
    args = parser.parse_args()

    args.hop_length = int(args.hop_length * SAMPLE_RATE)
    args.win_length = int(args.win_length * SAMPLE_RATE)
    logger.info(
        f"Adjusted Hoplength and window size:\nn_fft: {args.n_fft}\nwindow: {args.win_length}\nhop: {args.hop_length}"
    )
    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    data = pd.read_csv(TASKS[args.task]['data'],
                       header=None,
                       names=['filename'])
    label_df = pd.read_csv(TASKS[args.task]['label'], sep='\s+')
    logger.info(f"Label_df shape is {label_df.shape}")

    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_file=model_kwargs_pack['pretrained']).to(DEVICE).eval()
    encoder = torch.load(model_kwargs_pack['encoder'])
    ## VAD preprocessing data
    vad_label_helper_df = label_df.copy()
    vad_label_helper_df['onset'] = np.ceil(vad_label_helper_df['onset'] /
                                           model_resolution).astype(int)
    vad_label_helper_df['offset'] = np.ceil(vad_label_helper_df['offset'] /
                                            model_resolution).astype(int)

    vad_label_helper_df = vad_label_helper_df.groupby(['filename']).agg({
        'onset':
        tuple,
        'offset':
        tuple,
        'event_label':
        tuple
    }).reset_index()
    dset = OnlineLogMelDataset(data['filename'].values.tolist(),
                               hop_length=args.hop_length,
                               n_fft=args.n_fft,
                               win_length=args.win_length,
                               n_mels=args.n_mels)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=8,
                                          shuffle=False)
    logger.trace(model)

    output_dfs = []
    threshold = tuple(args.threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    speech_frame_predictions, speech_frame_ground_truth, speech_frame_prob_predictions = [], [],[]
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.threshold
    else:
        postprocessing_method = utils.double_threshold
    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        for feature, filename in dloader:
            feature = torch.as_tensor(feature).to(DEVICE)
            # PANNS output a dict instead of 2 values
            if 'cnn14' in args.model:
                out_dict = model(feature)
                prediction_tag = out_dict['clipwise_output'].to('cpu')
                prediction_time = out_dict['framewise_output'].to('cpu')
            # For the CRNN models
            else:
                prediction_tag, prediction_time = model(feature)
                prediction_tag = prediction_tag.to('cpu')
                prediction_time = prediction_time.to('cpu')

            if prediction_time is not None:  # Some models do not predict timestamps

                cur_filename = filename[
                    0] if not 'aurora' in args.task else Path(filename[0]).stem

                thresholded_prediction = postprocessing_method(
                    prediction_time, *threshold)

                ## VAD predictions
                speech_frame_prob_predictions.append(
                    prediction_time[..., speech_label_idx].squeeze())
                ### Thresholded speech predictions
                speech_prediction = thresholded_prediction[
                    ..., speech_label_idx].squeeze()
                speech_frame_predictions.append(speech_prediction)
                targets = vad_label_helper_df[vad_label_helper_df['filename']
                                              == cur_filename][[
                                                  'onset', 'offset'
                                              ]].values[0]
                target_arr = np.zeros_like(speech_prediction)
                for start, end in zip(*targets):
                    target_arr[start:end] = 1
                speech_frame_ground_truth.append(target_arr)

                #### SED predictions

                labelled_predictions = utils.decode_with_timestamps(
                    encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])
                if not pred_label_df.empty:
                    if 'aurora' in args.task:
                        pred_label_df['filename'] = cur_filename
                    else:
                        pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)

    full_prediction_df = pd.concat(output_dfs)
    prediction_df = full_prediction_df[full_prediction_df['event_label'] ==
                                       'Speech']
    assert set(['onset', 'offset', 'filename', 'event_label'
                ]).issubset(prediction_df.columns), "Format is wrong"
    assert set(['onset', 'offset', 'filename',
                'event_label']).issubset(label_df.columns), "Format is wrong"
    logger.info("Calculating VAD measures ... ")
    speech_frame_ground_truth = np.concatenate(speech_frame_ground_truth,
                                               axis=0)
    speech_frame_predictions = np.concatenate(speech_frame_predictions, axis=0)
    speech_frame_prob_predictions = np.concatenate(
        speech_frame_prob_predictions, axis=0)

    vad_results = []
    tn, fp, fn, tp = confusion_matrix(speech_frame_ground_truth,
                                      speech_frame_predictions).ravel()
    fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))

    auc = roc_auc_score(speech_frame_ground_truth,
                        speech_frame_prob_predictions) * 100
    for avgtype in ('micro', 'macro', 'binary'):
        precision, recall, f1, _ = precision_recall_fscore_support(
            speech_frame_ground_truth,
            speech_frame_predictions,
            average=avgtype)
        vad_results.append((avgtype, 100 * precision, 100 * recall, 100 * f1))

    logger.info("Calculating segment based metric .. ")
    # Change order just for better printing in file
    prediction_df = prediction_df[[
        'filename', 'onset', 'offset', 'event_label'
    ]]
    metric = segment_based_evaluation_df(label_df,
                                         prediction_df,
                                         time_resolution=args.time_resolution)
    logger.info("Calculating event based metric .. ")
    event_metric = event_based_evaluation_df(label_df, prediction_df)

    args.output_path = Path(
        args.output_path
    ) / args.task / args.model / args.pretrained_from / uuid.uuid1().hex
    args.output_path.mkdir(parents=True)
    prediction_df.to_csv(args.output_path / 'speech_predictions.tsv',
                         sep='\t',
                         index=False)
    full_prediction_df.to_csv(args.output_path / 'predictions.tsv',
                              sep='\t',
                              index=False)
    with open(args.output_path / 'evaluation.txt', 'w') as fp:
        print(vars(args), file=fp)
        print(metric, file=fp)
        print(event_metric, file=fp)
        for avgtype, precision, recall, f1 in vad_results:
            print(
                f"VAD {avgtype} F1: {f1:<10.3f} {precision:<10.3f} Recall: {recall:<10.3f}",
                file=fp)
        print(f"FER: {fer:.2f}", file=fp)
        print(f"AUC: {auc:.2f}", file=fp)
    logger.info(f"Results are at {args.output_path}")
    for avgtype, precision, recall, f1 in vad_results:
        print(
            f"VAD {avgtype:<10} F1: {f1:<10.3f} Pre: {precision:<10.3f} Recall: {recall:<10.3f}"
        )
    print(f"FER: {fer:.2f}")
    print(f"AUC: {auc:.2f}")
    print(event_metric)
    print(metric)


if __name__ == "__main__":
    main()
