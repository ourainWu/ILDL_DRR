from measures import *
from sklearn.model_selection import KFold
from metrics import *
from sklearn.preprocessing import scale, StandardScaler
from model import Model
import torch.optim as optim
import math

def train(train_feature, train_missing_y, train_mask):

    features_dim = len(train_feature[0])
    labels_dim = len(train_missing_y[0])

    # train_feature = scale(train_feature)
    x_data = torch.from_numpy(train_feature).float().to(device)
    y_distribution = torch.from_numpy(train_missing_y).float().to(device)
    train_mask = torch.from_numpy(train_mask).float().to(device)

    model = Model(features_dim, labels_dim)

    dataset = GetMissingDataset(x_data, y_distribution, train_mask)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    op = optim.Adam(list(model.parameters()), lr = args.lr, weight_decay=args.para_hyper[2])
    lr_s = torch.optim.lr_scheduler.StepLR(op, step_size=15, gamma=0.9)

    loss = []
    before_loss_all = math.inf

    for epoch_pre in range(args.epochs):
        temp_loss_all = 0
        for batch_idx, (ba_x, ba_y, ba_mask, idx) in enumerate(train_loader):
            model.train()
            batch_hat = model(ba_x)

            pre_loss = mask_loss(ba_y, batch_hat, ba_mask)
            pre_ranking_loss = ranking_loss(ba_y, batch_hat, ba_mask)
            pre_margin_loss = margin_loss(ba_y, batch_hat, ba_mask)
            pre_dpa_loss = dpa_mask_normalized(ba_y, batch_hat, ba_mask)

            loss_all = pre_loss + args.para_hyper[0] * pre_dpa_loss + args.para_hyper[1] * pre_margin_loss

            temp_loss_all += loss_all

            op.zero_grad()
            loss_all.backward()
            op.step()
            lr_s.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_pre, batch_idx * len(ba_y.T[0]), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss_all))

        print("++++++++++++++++++++++++++++++++++++++ now the whole loss is", temp_loss_all)
        loss.append(temp_loss_all.data.cpu().numpy().item())
        if torch.abs(before_loss_all - temp_loss_all) < 0.0001: ## 0.0001
            break
        before_loss_all = temp_loss_all

    print(loss)
    return model

def test(test_feature):
    # test_feature = scale(test_feature)
    x_data_test = torch.from_numpy(test_feature).float().to(device)
    predict_y = model(x_data_test)
    preds = []
    preds.extend(predict_y.data.cpu().numpy())

    return np.array(preds)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='Yeast_alpha', metavar='N',
                        help='the name of dataset [default: sample_data]')
    parser.add_argument('--obr', type=int, default=0.1, metavar='N',
                        help='the 10 percent labels in dataset are obeserved')
    parser.add_argument('--para_hyper', nargs=3, type=int, default=[0.00001, 1, 0.1], metavar='N',
                        help='input batch size for training [default: [1, 1]]')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training [default: 64]')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate [default: 5e-3]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='choose CUDA device [default: cuda:1]')
    parser.add_argument('--seed', '-seed', type=int, default=10,
                        help='random seed (default: 0)')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device(args.device if args.cuda else 'cpu')

    metric = []
    data_X, data_y, obrT = load_missing_dataset(args.dataname, args.obr)

    for i in range(1):
        train_test_indices = []
        KF = KFold(n_splits=10, random_state=42, shuffle=True)
        for train_index, test_index in KF.split(data_X):
            train_feature = data_X[train_index]
            test_feature = data_X[test_index]

            train_y = data_y[train_index]
            train_mask = obrT[train_index]
            test_y = data_y[test_index]

            train_missing_y = np.where(train_mask == 1, train_y, 0)

            # scaler = StandardScaler()
            # train_feature = scaler.fit_transform(train_feature)
            # test_feature = scaler.transform(test_feature)

            model = train(train_feature, train_missing_y, train_mask)
            predict_y = test(test_feature)

            dists = [
                chebyshev(test_y, predict_y),
                clark(test_y, predict_y),
                canberra(test_y, predict_y),
                kl_divergence(test_y, predict_y),
                cosine(test_y, predict_y),
                intersection(test_y, predict_y),
                spearman(test_y, predict_y),
                kendall(test_y, predict_y)
                    ]

            print(dists)
            metric.append(dists)

    print(metric)
    for i in range(8):
        print("%0.4f±%0.4f" % (np.array(metric)[:, i].mean(), np.array(metric)[:, i].std()))

    # name = f"results/Method_NO(3407)+{args.dataname}+{args.obr}+{args.para_hyper}.npy"
    name = f"results/ILDL_DRR+{args.dataname}+{args.obr}+{args.para_hyper}.npy"
    # name = f"results_incom/Method_NO(3407)+{args.dataname}+{args.obr}+{args.para_hyper}.npy"
    # np.save(name, np.array(metric))