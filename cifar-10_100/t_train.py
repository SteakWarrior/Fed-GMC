import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from copy import deepcopy
import torch
from torchvision import models as models
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import model_test
import os
import time
import torch.nn.functional as F
import my_resnet
import Gen_noniid
import user_model
import random

class Arguments():
    def __init__(self):
        self.batch_size = 128
        self.test_batch_size = 1000
        self.server_epochs = 1
        self.user_epochs = 5
        self.round = 500
        self.lr = 1e-2
        self.momentum = 0.9
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 20
        self.save_model = False
        self.T = 6
        self.alpha = 0.5
        self.path = r'/home/zsh/flowerpot/security/results'
        self.best_acc = [0 for _ in range(15)]
        self.teacher_pred = 0
        self.student_pred = 0
        self.best_acc1 = [0 for _ in range(15)]
        self.best_acc2 = [0 for _ in range(15)]
        self.sample_cnt = 3
        self.user_list = [i for i in range(15)]


def train(args, device, model, criterion, train_loader, optimizer, epoch, round,  server_opt, server_model):

    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)

        for epoch in range(1, args.server_epochs + 1):
            server_train(args, server_opt, server_model, model.semi_f.detach(), label, round, epoch, criterion, output)

        logit_s = args.teacher_pred
        logit_stu = args.student_pred

        hard_loss = F.cross_entropy(output, label)

        soft_loss = criterion(F.log_softmax(output / args.T, dim=1), F.softmax(logit_s.detach() / args.T, dim=1))
        offset_loss = criterion(F.log_softmax(output, dim=1), F.softmax(logit_stu.detach(), dim=1))

        loss = hard_loss + offset_loss + soft_loss
        loss.backward()
        optimizer.step()



def server_train(args, server_opt, server_model, data, label, round, epoch, criterion, js_dis):
    model1, model2 = server_model
    optimizer1, optimizer2 = server_opt
    model1.train()
    model2.train()

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    output2 = model2(data)
    output1 = model1(data)

    if epoch == 1:
        args.teacher_pred = output1
        args.student_pred = output2

    hard_loss1 = F.cross_entropy(output1, label)
    hard_loss1.backward()
    optimizer1.step()

    hard_loss2 = F.cross_entropy(output2, label)
    soft_loss = criterion(F.log_softmax(output2 / args.T, dim=1), F.softmax(output1.detach() / args.T, dim=1))
    loss = hard_loss2 + (args.T ** 2) * soft_loss
    loss.backward()
    optimizer2.step()

    #with torch.no_grad():
    #    if((round + 1) % 100 == 0 and epoch == 1):
    #        M = (F.softmax(js_dis, dim=1) + F.softmax(output1.detach(), dim=1)) / 2
    #        dd = 0.5 * criterion(F.log_softmax(js_dis, dim=1), M) + \
    #                                            0.5 * criterion(F.log_softmax(output1.detach(), dim=1), M)
    #        with open(args.path + r'/' + 'JS_distance.txt', 'a') as f:
    #            f.write('{}\n'.format(dd))


def server_test(args, server_model, test_loader, ex_layer, round, user_id, epoch):
    model1, model2 = server_model
    model1.eval()
    model2.eval()

    test_loss1 = 0
    test_loss2 = 0
    total_loss1 = 0
    total_loss2 = 0
    acc1 = 0
    acc2 = 0

    conv1, bn1 = ex_layer

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.cuda(), label.cuda()
            feature = F.relu(bn1(conv1(data)))
            output2 = model2(feature)
            output1 = model1(feature)
            predicted1 = torch.max(output1, 1)[1].data.cpu().numpy()
            predicted2 = torch.max(output2, 1)[1].data.cpu().numpy()
            test_loss1 = F.cross_entropy(output1, label)
            test_loss2 = F.cross_entropy(output2, label)
            total_loss1 += test_loss1.data.item()
            total_loss2 += test_loss2.data.item()
            acc1 += (predicted1 == label.cpu().data.numpy()).astype(int).sum().item()
            acc2 += (predicted2 == label.cpu().data.numpy()).astype(int).sum().item()
        acc_1 = 100. * acc1 / (len(test_loader) * args.test_batch_size)
        acc_2 = 100. * acc2 / (len(test_loader) * args.test_batch_size)

        print("Epoch:{} Stu_{} accuracy: {:.1f}%\t loss: {}".format(epoch, user_id, acc_2, total_loss2 / len(test_loader)))
        print("Epoch:{} Server accuracy: {:.1f}%\t loss: {}".format(epoch, acc_1, total_loss1 / len(test_loader)))

        with open(args.path + r'/' + 'stu_{}.txt'.format(user_id), 'a') as f:
            f.write('{}%\n'.format(acc_2))
        
        with open(args.path + r'/' + 'ResNet51_{}.txt'.format(user_id), 'a') as f:
            f.write('{}%\n'.format(acc_1))

        if acc_1 > args.best_acc1[user_id]:
            args.best_acc1[user_id] = acc_1
        
        if acc_2 > args.best_acc2[user_id]:
            args.best_acc2[user_id] = acc_2



def test(args, device, model, test_loader, user_id, criterion, udp_socket, round):
    model.eval()
    test_loss = 0
    total_loss = 0
    acc = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            predicted = torch.max(output, 1)[1].data.cpu().numpy()
            test_loss = F.cross_entropy(output, label)
            total_loss += test_loss.data.item()
            acc += (predicted == label.cpu().data.numpy()).astype(int).sum().item()
            acc_1 = 100. * acc / (len(test_loader) * args.test_batch_size)

        print("Round: {}\nUser_{}: {:.1f}%\t loss: {}".format(round, user_id, acc_1, total_loss / len(test_loader)))

        with open(args.path + r'/' + 'user_{}.txt'.format(user_id), 'a') as f:
            f.write('{}%\n'.format(acc_1))

        #with open(args.path + r'/' + 'user_loss_{}.txt'.format(user_id), 'a') as f:
        #    f.write('{}%\n'.format(total_loss / len(test_loader)))

        if acc_1 > args.best_acc[user_id]:
            args.best_acc[user_id] = acc_1


def train_main(train_loaders, test_loader, args, server_model, model_list, device, criterion, server_opt):
    for round in range(1, args.round + 1):
        # ls = random.sample(args.user_list, args.sample_cnt)
        for i in range(15):
            if(round % 10 == 0):
                ori_model = model_list[i].state_dict()
                ori_model.update(deepcopy(server_model[1].state_dict()))
                model_list[i].load_state_dict(ori_model)
           	
            opt = optim.SGD(model_list[i].parameters(), lr=args.lr, weight_decay=5e-4)

            for epoch in range(1, args.user_epochs + 1):
                train(args, device, model_list[i], criterion, train_loaders[i], opt, epoch, round, server_opt, server_model)
        # for i in range(15):
            test(args, device, model_list[i], test_loader, i, round, epoch, round)
            ex_layer = (model_list[i].conv1, model_list[i].bn1)
            server_test(args, server_model, test_loader, ex_layer, round, i, epoch)


def main():
    t_start = time.time()

    #------------------------------------------------------------------#

    args = Arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #-----------------------------------------------------------------#

    model_list = []
    for i in range(15):
        model = user_model.ResNet10().to(device)
        model_list.append(model)

    teacher_model = my_resnet.ResNet51().to(device)
    student_model = model_test.ResNet10().to(device)

    server_opt = []

    server_opt.append(optim.SGD(teacher_model.parameters(), lr=args.lr, weight_decay=5e-4))
    server_opt.append(optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=5e-4))

    criterion = nn.KLDivLoss().to(device)

    train_loaders, test_loader = Gen_noniid.get_user_dataset()

    train_main(train_loaders, test_loader, args, (teacher_model, student_model), model_list, device, criterion, server_opt)

    t_end = time.time()
    print('Train cost: {}'.format(t_end - t_start))
    
    for i in range(len(args.best_acc)):
        print("user_{}: {:.1f}%".format(i, args.best_acc[i]))

    torch.save(model_list[0], args.path)
    
    for i in range(len(args.best_acc1)):
        print("teacher_{}: {:.1f}%".format(i, args.best_acc1[i]))
    
    torch.save(student_model, args.path)
    
    for i in range(len(args.best_acc2)):
        print("stu_{}: {:.1f}%".format(i, args.best_acc2[i]))

    torch.save(teacher_model, args.path)


if __name__ == '__main__':
    main()

