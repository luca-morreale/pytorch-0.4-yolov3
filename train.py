from __future__ import print_function
import sys

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import gc

import dataset
from utils import *
from image import correct_yolo_boxes
from cfg import parse_cfg
from darknet import Darknet
import argparse

from tensorboardX import SummaryWriter

FLAGS = None
unparsed = None
device = None

# global variables
# Training settings
# Train parameters
use_cuda      = None
eps           = 1e-5
keep_backup   = 10
save_interval = 1000  # epoches
test_interval = 10  # epoches
dot_interval  = 70  # batches

# Test parameters
evaluate = False
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

# no test evalulation
no_eval = False
init_eval = False

# Training settings
def load_testlist(testlist):
    init_width = model.width
    init_height = model.height

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)
    return loader

def main():
    datacfg    = FLAGS.data
    cfgfile    = FLAGS.config
    weightfile = FLAGS.weights
    no_eval    = FLAGS.no_eval
    init_eval  = FLAGS.init_eval

    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(cfgfile)[0]

    global use_cuda
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)

    globals()["trainlist"]     = data_options['train']
    globals()["testlist"]      = data_options['valid']
    globals()["backupdir"]     = data_options['backup']
    globals()["gpus"]          = data_options['gpus']  # e.g. 0,1,2,3
    globals()["ngpus"]         = len(gpus.split(','))
    globals()["num_workers"]   = int(data_options['num_workers'])

    globals()["batch_size"]    = int(net_options['batch'])
    globals()["max_batches"]   = int(net_options['max_batches'])
    globals()["learning_rate"] = float(net_options['learning_rate'])
    globals()["momentum"]      = float(net_options['momentum'])
    globals()["decay"]         = float(net_options['decay'])
    globals()["steps"]         = [float(step) for step in net_options['steps'].split(',')]
    globals()["scales"]        = [float(scale) for scale in net_options['scales'].split(',')]
    globals()['n_boxes']       = 10

    if FLAGS.backup_dir is not None:
        globals()['backupdir'] = FLAGS.backup_dir

    print(backupdir)
    #Train parameters
    global max_epochs
    max_epochs = 1000
    # try:
    #     max_epochs = int(net_options['max_epochs'])
    # except KeyError:
    #     nsamples = file_lines(trainlist)
    #     max_epochs = (max_batches*batch_size)//nsamples+1

    seed = 0 # int(time.time())
    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    global model
    model = Darknet(cfgfile, use_cuda=use_cuda)
    if weightfile is not None:
        model.load_weights(weightfile)

    monitor = SummaryWriter()

    #model.print_network()

    nsamples = file_lines(trainlist)
    #initialize the model
    if FLAGS.reset:
        model.seen = 0

    global loss_layers
    loss_layers = model.loss_layers
    for l in loss_layers:
        l.seen = model.seen

    globals()["test_loader"] = load_testlist(testlist)
    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay*batch_size}]
    global optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size)
    # optimizer = optim.SGD(model.parameters(),
    #                     lr=learning_rate/batch_size, momentum=momentum,
    #                     dampening=0, weight_decay=decay*batch_size)

    if evaluate:
        logging('evaluating ...')
        test(0)
    else:
        try:
            print("Training for ({:d},{:d})".format(0, max_epochs))
            fscore = 0
            if init_eval and not no_eval :
                print('>> initial evaluating ...')
                mfscore = test(0)
                print('>> done evaluation.')
            else:
                curmodel().seen = 0
                mfscore = 0.5
            for epoch in range(max_epochs):
                print(epoch)
                nsamples = train(epoch, monitor)
                # if epoch % save_interval == 0:
                #     savemodel(epoch, nsamples)
                if not no_eval and epoch >= test_interval and (epoch%test_interval) == 0:
                    print('>> intermittent evaluating ...')
                    fscore = test(epoch)
                    print('>> done evaluation.')
                if FLAGS.localmax and fscore > mfscore:
                    mfscore = fscore
                    savemodel(epoch * nsamples, True)
                print('-'*90)
        except KeyboardInterrupt:
            print('='*80)
            print('Exiting from training by interrupt')

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def curmodel():
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    return cur_model

def train(epoch, monitor):
    global processed_batches, learning_rate
    t0 = time.time()
    cur_model = curmodel()
    init_width = cur_model.width
    init_height = cur_model.height
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                        ]),
                        train=True,
                        seen=cur_model.seen,
                        batch_size=batch_size,
                        num_workers=num_workers, n_boxes=n_boxes),
                        batch_size=batch_size, shuffle=False, **kwargs)

    processed_batches = cur_model.seen//batch_size

    print('current learning rate {}'.format(learning_rate))
    lr = adjust_learning_rate(optimizer, processed_batches)
    print('new learnning rate {}'.format(lr))

    logging('[%03d] processed %d samples, lr %e' % (epoch, cur_model.seen, lr))
    model.train()
    t1 = time.time()

    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        cur_model.seen += 1
        t3 = time.time()
        data, target = data.to(device), target.to(device)

        t4 = time.time()
        optimizer.zero_grad()

        t5 = time.time()
        output = model(data)

        t6 = time.time()
        org_loss   = []
        cls_loss   = []
        conf_loss  = []
        coord_loss = []
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            # ol=l(output[i]['x'], target)
            coord, conf, cls, ol = l(output[i]['x'], target)
            org_loss.append(ol)
            cls_loss.append(cls)
            conf_loss.append(conf)
            coord_loss.append(coord)

        t7 = time.time()

        #for i, l in enumerate(reversed(org_loss)):
        #    l.backward(retain_graph=True if i < len(org_loss)-1 else False)
        # org_loss.reverse()
        sum(org_loss).backward()
        monitor.add_scalar('yolov3/loss', sum(org_loss), cur_model.seen)
        monitor.add_scalar('yolov3/coord_loss', sum(coord_loss), cur_model.seen)
        monitor.add_scalar('yolov3/conf_loss', sum(conf_loss), cur_model.seen)
        monitor.add_scalar('yolov3/cls_loss', sum(cls_loss), cur_model.seen)
        # monitor.add_scalar('yolov3/lr', lr, cur_model.seen)

        nn.utils.clip_grad_norm_(model.parameters(), 10000)
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)

        t8 = time.time()
        optimizer.step()

        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
        del data, target
        org_loss.clear()
        gc.collect()

        if cur_model.seen % save_interval == 0:
            savemodel(cur_model.seen)

    print('')
    t1 = time.time()
    nsamples = len(train_loader.dataset)
    logging('[%03d] training with %f samples/s' % (epoch, nsamples/(t1-t0)))
    return nsamples

def savemodel(n_iter, curmax=False):
    cur_model = curmodel()
    if curmax:
        logging('save local maximum weights to %s/localmax.weights' % (backupdir))
    else:
        logging('save weights to %s/%d.weights' % (backupdir, n_iter))
    # cur_model.seen = n_iter
    if curmax:
        cur_model.save_weights('%s/localmax.weights' % (backupdir))
    else:
        cur_model.save_weights('%s/%06d.weights' % (backupdir, n_iter))
        old_wgts = '%s/%06d.weights' % (backupdir, n_iter-keep_backup*save_interval)

def test(epoch):
    def truths_length(truths):
        for i in range(n_boxes):
            if truths[i][1] == 0:
                return i
        return n_boxes

    model.eval()
    cur_model = curmodel()
    num_classes = cur_model.num_classes
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    if cur_model.net_name() == 'region': # region_layer
        shape=(0,0)
    else:
        shape=(cur_model.width, cur_model.height)
    with torch.no_grad():
        for data, target, org_w, org_h in test_loader:
            data = data.to(device)
            output = model(data)
            all_boxes = get_all_boxes(output, shape, conf_thresh, num_classes, use_cuda=use_cuda)

            for k in range(len(all_boxes)):
                boxes = all_boxes[k]
                correct_yolo_boxes(boxes, org_w[k], org_h[k], cur_model.width, cur_model.height)
                boxes = np.array(nms(boxes, nms_thresh))

                truths = target[k].view(-1, 5)
                num_gts = truths_length(truths)
                total = total + num_gts
                num_pred = len(boxes)
                if num_pred == 0:
                    continue

                proposals += int((boxes[:,4]>conf_thresh).sum())
                for i in range(num_gts):
                    gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                    gt_boxes = gt_boxes.repeat(num_pred,1).t()
                    pred_boxes = torch.FloatTensor(boxes).t()
                    best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                    # pred_boxes and gt_boxes are transposed for torch.max
                    if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                        correct += 1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    savelog("[%03d] correct: %d, precision: %f, recall: %f, fscore: %f" % (epoch, correct, precision, recall, fscore))
    return fscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d',
        type=str, required=True, help='data definition file')
    parser.add_argument('--config', '-c',
        type=str, required=True, help='network configuration file')
    parser.add_argument('--weights', '-w',
        type=str, required=True, help='initial weights file')
    parser.add_argument('--initeval', '-i', dest='init_eval', action='store_true',
        help='performs inital evalulation')
    parser.add_argument('--noeval', '-n', dest='no_eval', action='store_true',
        help='prohibit test evalulation')
    parser.add_argument('--reset', '-r',
        action="store_true", default=False, help='initialize the epoch and model seen value')
    parser.add_argument('--localmax', '-l',
        action="store_true", default=False, help='save net weights for local maximum fscore')
    parser.add_argument('--backup_dir', '-bd', default=None, type=str)

    FLAGS, _ = parser.parse_known_args()
    main()

