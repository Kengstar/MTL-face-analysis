import torch


def normalized_errors(output, target, nr_of_landmarks):
    # first 2 landmarks are eyes
        left_eye_x = target[:, 0]
        left_eye_y = target[:, 1]
        right_eye_x =target[:, 2]
        right_eye_y = target[:, 3]
        iod = torch.sqrt(((left_eye_x - right_eye_x)**2) + ((left_eye_y - right_eye_y)**2))
        error = torch.nn.PairwiseDistance(p=2)(output, target) / nr_of_landmarks
        return error / iod


def failure_rate(errors):
    return torch.sum(errors > 0.1)


def iou(output, labels):
    pred = torch.where(output < 0.5, torch.tensor(0, dtype=torch.int), torch.tensor(1, dtype=torch.int))
    labels = labels.int()
    pred = pred.squeeze(1)
    labels = labels.squeeze(1)

    intersection = pred & labels
    intersection = torch.sum(torch.sum(intersection, dim=1), dim=1)

    union = pred | labels ## all pixel where 1.0
    union = torch.sum(torch.sum(union, dim=1), dim=1)
    return intersection.float() / union.float()


